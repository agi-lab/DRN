import os


os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import numpy as np
import torch
import torch.nn as nn
from synthetic_dataset import generate_synthetic_tensordataset
from synthetic_dataset import generate_synthetic_data
from drn import GLM, CANN, MDN, DDR, DRN, train, ddr_cutpoints, merge_cutpoints

# force CPU in .fit
FIT_KW = dict(
    accelerator="cpu", devices=1, deterministic=True, enable_checkpointing=False
)


def _extract_numpy_from_tensordataset(ds):
    # ds is a torch.utils.data.TensorDataset
    X_t, y_t = ds.tensors
    return X_t.cpu().numpy(), y_t.cpu().numpy()


def _compare_params(m1, m2, atol=1e-6, ignore_dispersion=True):
    for (n1, p1), (n2, p2) in zip(m1.named_parameters(), m2.named_parameters()):
        assert n1 == n2, "Parameter lists aren’t aligned!"
        if "dispersion" in n1 and ignore_dispersion:
            continue

        close = torch.allclose(p1, p2, atol=atol)
        both_nan = torch.isnan(p1).all() and torch.isnan(p2).all()
        assert (
            close or both_nan
        ), f"Parameters are not close enough: {p1} vs {p2} (close={close}, both_nan={both_nan})"


def test_glm_train_vs_fit_equivalence():
    """GLM trained via `train(...)` vs. via `.fit(...)` should end up identical."""
    seed = 42
    # get torch tensors + datasets
    X_t, y_t, train_ds, val_ds = generate_synthetic_tensordataset()
    # extract numpy arrays for .fit
    X_train_np, y_train_np = _extract_numpy_from_tensordataset(train_ds)
    X_val_np, y_val_np = _extract_numpy_from_tensordataset(val_ds)

    # 1) train(...) version
    torch.manual_seed(seed)
    glm1 = GLM("gamma")
    train(glm1, train_ds, val_ds, epochs=10)

    # 2) .fit(...) version
    torch.manual_seed(seed)
    glm2 = GLM("gamma")
    glm2.fit(
        X_train_np,
        y_train_np,
        X_val_np,
        y_val_np,
        grad_descent=True,
        epochs=10,
        **FIT_KW,
    )

    # compare every learned parameter
    _compare_params(glm1, glm2, atol=1e-2)


def test_cann_train_vs_fit_equivalence():
    """CANN trained via `train(...)` vs. via `.fit(...)` should end up identical."""
    seed = 1234
    # get torch tensors + datasets
    X_t, y_t, train_ds, val_ds = generate_synthetic_tensordataset()
    # extract numpy arrays for .fit
    X_train_np, y_train_np = X_t.cpu().numpy(), y_t.cpu().numpy()
    X_val_np, y_val_np = _extract_numpy_from_tensordataset(val_ds)

    # 1) train(...) version
    torch.manual_seed(seed)
    glm1 = GLM("gamma")
    glm1.fit(X_train_np, y_train_np)

    x_obs = torch.tensor(X_train_np[:1], dtype=torch.float32)

    cann1 = CANN(glm1, num_hidden_layers=1, hidden_size=16)
    _ = cann1(x_obs)  # trigger LazyLinear init
    train(cann1, train_ds, val_ds, epochs=1)

    # 2) .fit(...) version
    torch.manual_seed(seed)
    glm2 = GLM("gamma")
    glm2.fit(X_train_np, y_train_np)

    cann2 = CANN(glm2, num_hidden_layers=1, hidden_size=16)

    _ = cann2(x_obs)  # trigger LazyLinear init
    cann2.fit(X_train_np, y_train_np, X_val_np, y_val_np, epochs=1, **FIT_KW)

    # compare every learned parameter
    _compare_params(cann1, cann2)


def test_mdn_train_vs_fit_equivalence():
    """MDN trained via `train(...)` vs. via `.fit(...)` should end up identical."""
    seed = 2025
    X_t, y_t, train_ds, val_ds = generate_synthetic_tensordataset()
    X_train_np, y_train_np = X_t.cpu().numpy(), y_t.cpu().numpy()
    X_val_np, y_val_np = _extract_numpy_from_tensordataset(val_ds)

    x_obs = torch.tensor(X_train_np[:1], dtype=torch.float32)

    # 1) train(...) version
    torch.manual_seed(seed)
    mdn1 = MDN("gamma", num_components=4)
    _ = mdn1(x_obs)  # trigger LazyLinear init
    train(mdn1, train_ds, val_ds, epochs=1)

    # 2) .fit(...) version
    torch.manual_seed(seed)
    mdn2 = MDN("gamma", num_components=4)
    _ = mdn2(x_obs)  # trigger LazyLinear init
    mdn2.fit(X_train_np, y_train_np, X_val_np, y_val_np, epochs=1, **FIT_KW)

    _compare_params(mdn1, mdn2)


def test_glm_train_vs_fit_early_stopping_equivalence():
    """GLM: train(...) vs .fit(...) with early stopping should still match."""
    seed = 42
    # get torch tensors + datasets
    X_t, y_t, train_ds, val_ds = generate_synthetic_tensordataset()
    X_train_np, y_train_np = _extract_numpy_from_tensordataset(train_ds)
    X_val_np, y_val_np = _extract_numpy_from_tensordataset(val_ds)

    # 1) train(...) version with patience=1
    torch.manual_seed(seed)
    glm1 = GLM("gamma")
    train(glm1, train_ds, val_ds, epochs=10, patience=3)

    # 2) .fit(...) version with the same epochs & patience
    torch.manual_seed(seed)
    glm2 = GLM("gamma")
    glm2.fit(
        X_train_np,
        y_train_np,
        X_val_np,
        y_val_np,
        grad_descent=True,
        epochs=10,
        patience=3,
        **FIT_KW,
    )

    # compare every learned parameter
    _compare_params(glm1, glm2, atol=1e-2)


def test_cann_train_vs_fit_equivalence_ignoring_dispersion():
    """CANN trained via `train(...)` vs. via `.fit(...)` should end up identical."""
    seed = 1234
    # get torch tensors + datasets
    X_t, y_t, train_ds, val_ds = generate_synthetic_tensordataset()
    # extract numpy arrays for .fit
    X_train_np, y_train_np = X_t.cpu().numpy(), y_t.cpu().numpy()
    X_val_np, y_val_np = _extract_numpy_from_tensordataset(val_ds)

    # 1) train(...) version
    torch.manual_seed(seed)
    glm1 = GLM("gamma")
    glm1.fit(X_train_np, y_train_np)

    x_obs = torch.tensor(X_train_np[:1], dtype=torch.float32)

    cann1 = CANN(glm1, num_hidden_layers=1, hidden_size=16)
    _ = cann1(x_obs)  # trigger LazyLinear init
    train(cann1, train_ds, val_ds, epochs=1)

    # 2) .fit(...) version
    torch.manual_seed(seed)
    glm2 = GLM("gamma")
    glm2.fit(X_train_np, y_train_np)

    cann2 = CANN(glm2, num_hidden_layers=1, hidden_size=16)
    _ = cann2(x_obs)  # trigger LazyLinear init
    cann2.fit(X_train_np, y_train_np, X_val_np, y_val_np, epochs=1, **FIT_KW)

    # Add a fake dispersion parameter to cann1
    cann1.dispersion = nn.Parameter(torch.Tensor([1234.5]))

    # Make sure the test still passes by comparing the non-dispersion parameters
    _compare_params(cann1, cann2, ignore_dispersion=True)

    # Update dispersion for cann1 to match cann2 (which gets the dispersion from model.fit)
    cann1.update_dispersion(X_t, y_t)

    _compare_params(cann1, cann2, ignore_dispersion=False)


def test_mdn_train_vs_fit_early_stopping_equivalence():
    """MDN: train(...) vs .fit(...) with early stopping should match."""
    seed = 2025
    X_t, y_t, train_ds, val_ds = generate_synthetic_tensordataset()
    # extract numpy arrays for .fit
    X_train_np, y_train_np = _extract_numpy_from_tensordataset(train_ds)
    X_val_np, y_val_np = _extract_numpy_from_tensordataset(val_ds)

    x_obs = torch.tensor(X_train_np[:1], dtype=torch.float32)

    # 1) train(...) version with patience=2
    torch.manual_seed(seed)
    mdn1 = MDN("gamma", num_components=4)
    _ = mdn1(x_obs)  # trigger LazyLinear init
    train(mdn1, train_ds, val_ds, epochs=10, patience=2)

    # 2) .fit(...) version with the same epochs & patience
    torch.manual_seed(seed)
    mdn2 = MDN("gamma", num_components=4)
    _ = mdn2(x_obs)  # trigger LazyLinear init
    mdn2.fit(
        X_train_np, y_train_np, X_val_np, y_val_np, epochs=10, patience=2, **FIT_KW
    )

    # compare parameters
    _compare_params(mdn1, mdn2)


def test_ddr_train_vs_fit_equivalence():
    """DDR: train(...) vs .fit(...) should end up identical."""
    seed = 31415
    X_t, y_t, train_ds, val_ds = generate_synthetic_tensordataset()
    X_train_np, y_train_np = _extract_numpy_from_tensordataset(train_ds)
    X_val_np, y_val_np = _extract_numpy_from_tensordataset(val_ds)

    # compute cutpoints automatically
    c0 = min(y_train_np.min() * 1.05, 0)
    cK = y_train_np.max() * 1.05
    cps = ddr_cutpoints(c0, cK, proportion=0.1, n=len(y_train_np))

    x_obs = torch.tensor(X_train_np[:1], dtype=torch.float32)

    # 1) train(...) version
    torch.manual_seed(seed)
    ddr1 = DDR(cutpoints=cps, hidden_size=32)
    _ = ddr1(x_obs)  # trigger LazyLinear init
    train(ddr1, train_ds, val_ds, epochs=2)

    # 2) .fit(...) version
    torch.manual_seed(seed)
    ddr2 = DDR(cutpoints=cps, hidden_size=32)
    _ = ddr2(x_obs)  # trigger LazyLinear init
    ddr2.fit(X_train_np, y_train_np, X_val_np, y_val_np, epochs=2, **FIT_KW)

    # compare parameters
    _compare_params(ddr1, ddr2)


def test_drn_train_vs_fit_equivalence():
    """DRN: train(...) vs .fit(...) should end up identical (ignoring dispersion)."""
    seed = 2718
    X_t, y_t, train_ds, val_ds = generate_synthetic_tensordataset()
    X_train_np, y_train_np = _extract_numpy_from_tensordataset(train_ds)
    X_val_np, y_val_np = _extract_numpy_from_tensordataset(val_ds)

    # prepare base GLM and estimate dispersion
    torch.manual_seed(seed)
    glm1 = GLM("gamma")
    train(glm1, train_ds, val_ds, epochs=2)
    glm1.update_dispersion(X_t, y_t)

    torch.manual_seed(seed)
    glm2 = GLM("gamma")
    train(glm2, train_ds, val_ds, epochs=2)
    glm2.update_dispersion(X_t, y_t)

    # generate DRN cutpoints
    c0 = min(y_train_np.min() * 1.05, 0)
    cK = y_train_np.max() * 1.05
    base_cps = ddr_cutpoints(c0, cK, proportion=0.1, n=len(y_train_np))
    drn_cps = merge_cutpoints(base_cps, y_train_np, min_obs=2)

    x_obs = torch.tensor(X_train_np[:1], dtype=torch.float32)

    # 1) train(...) version
    torch.manual_seed(seed)
    drn1 = DRN(glm1, cutpoints=drn_cps, hidden_size=32)
    _ = drn1(x_obs)  # trigger LazyLinear init
    train(drn1, train_ds, val_ds, epochs=2)

    # 2) .fit(...) version
    torch.manual_seed(seed)
    drn2 = DRN(glm2, cutpoints=drn_cps, hidden_size=32)
    _ = drn2(x_obs)  # trigger LazyLinear init
    drn2.fit(X_train_np, y_train_np, X_val_np, y_val_np, epochs=2, **FIT_KW)

    # compare parameters (ignore dispersion parameter)
    _compare_params(drn1, drn2, ignore_dispersion=True)
