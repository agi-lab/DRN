import random
import numpy as np
import torch
from torch.utils.data import TensorDataset
from drn import (
    CANN,
    DDR,
    DRN,
    MDN,
    crps,
    ddr_cutpoints,
    ddr_loss,
    drn_cutpoints,
    drn_loss,
    gamma_deviance_loss,
    gamma_mdn_loss,
    gaussian_deviance_loss,
    gaussian_mdn_loss,
    train,
)


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def compute_crps(
    model: torch.nn.Module,
    X_val: torch.Tensor,
    Y_val: torch.Tensor,
    y_train_max: float,
    device: torch.device,
) -> float:
    """Compute the mean CRPS of model predictions on validation data."""
    grid_size = 3000
    grid = torch.linspace(0, y_train_max * 1.1, grid_size).unsqueeze(-1).to(device)

    model.eval()
    with torch.no_grad():
        dists = model.distributions(X_val)
        cdfs = dists.cdf(grid)
        grid = grid.squeeze()
        crps_score = crps(Y_val, grid, cdfs).mean().item()

    return crps_score


def objective_cann(
    params,
    glm,
    distribution,
    X_train: torch.Tensor,
    Y_train: torch.Tensor,
    X_val: torch.Tensor,
    Y_val: torch.Tensor,
    device: torch.device,
    patience: int,
):
    num_hidden_layers, hidden_size, dropout_rate, lr, batch_size = params

    cann = CANN(
        glm,
        num_hidden_layers=num_hidden_layers,
        hidden_size=hidden_size,
        dropout_rate=dropout_rate,
    )

    try:
        train(
            cann,
            (
                gaussian_deviance_loss
                if distribution == "gaussian"
                else gamma_deviance_loss
            ),
            TensorDataset(X_train, Y_train),
            TensorDataset(X_val, Y_val),
            epochs=2000,
            lr=lr,
            patience=patience,
            batch_size=int(batch_size),
            device=device,
        )

        cann.update_dispersion(X_train, Y_train)
        if not torch.isfinite(cann.dispersion):
            raise ValueError("Dispersion is not finite")

    except Exception as e:
        print(f"Training failed: {e}")
        return 1e6

    return compute_crps(cann, X_val, Y_val, Y_train.max().item(), device)


def objective_mdn(
    params,
    distribution,
    X_train: torch.Tensor,
    Y_train: torch.Tensor,
    X_val: torch.Tensor,
    Y_val: torch.Tensor,
    device: torch.device,
    patience: int,
):
    num_hidden_layers, hidden_size, dropout_rate, lr, num_components, batch_size = (
        params
    )

    mdn = MDN(
        X_train.shape[1],
        num_components=num_components,
        hidden_size=hidden_size,
        num_hidden_layers=num_hidden_layers,
        dropout_rate=dropout_rate,
        distribution=distribution,
    )

    try:
        train(
            mdn,
            gaussian_mdn_loss if distribution == "gaussian" else gamma_mdn_loss,
            TensorDataset(X_train, Y_train),
            TensorDataset(X_val, Y_val),
            lr=lr,
            batch_size=int(batch_size),
            epochs=2000,
            patience=patience,
            device=device,
        )
        mdn.eval()

    except Exception as e:
        print(e)
        print(f"Training failed: {e}")
        return 1e6

    crps_score = compute_crps(mdn, X_val, Y_val, Y_train.max().item(), device)

    return crps_score


def objective_ddr(
    params,
    X_train: torch.Tensor,
    Y_train: torch.Tensor,
    X_val: torch.Tensor,
    Y_val: torch.Tensor,
    device: torch.device,
    patience: int,
):
    num_hidden_layers, hidden_size, dropout_rate, lr, proportion, batch_size = params

    cutpoints = ddr_cutpoints(
        c_0=max(Y_train.min().item() * 1.05, 0),
        c_K=Y_train.max().item() * 1.05,
        proportion=proportion,
        n=X_train.shape[0],
    )

    ddr_model = DDR(
        X_train.shape[1],
        cutpoints,
        num_hidden_layers=num_hidden_layers,
        hidden_size=hidden_size,
        dropout_rate=dropout_rate,
    )

    try:
        train(
            ddr_model,
            ddr_loss,
            TensorDataset(X_train, Y_train),
            TensorDataset(X_val, Y_val),
            epochs=2000,
            lr=lr,
            patience=patience,
            batch_size=int(batch_size),
            device=device,
        )
    except Exception as e:
        print(f"Training failed: {e}")
        return 1e6

    crps_score = compute_crps(ddr_model, X_val, Y_val, Y_train.max().item(), device)
    return crps_score


def objective_drn(
    params,
    glm,
    kl_direction,
    criteria,
    X_train: torch.Tensor,
    Y_train: torch.Tensor,
    X_val: torch.Tensor,
    Y_val: torch.Tensor,
    device: torch.device,
    patience: int,
):
    (
        num_hidden_layers,
        hidden_size,
        dropout_rate,
        lr,
        kl_alpha,
        mean_alpha,
        dv_alpha,
        batch_size,
        proportion,
        min_obs,
    ) = params

    cutpoints = drn_cutpoints(
        c_0=max(Y_train.min().item() * 1.05, 0),
        c_K=Y_train.max().item() * 1.05,
        y=Y_train.cpu().numpy(),
        proportion=proportion,
        min_obs=min_obs,
    )

    drn_model = DRN(
        num_features=X_train.shape[1],
        cutpoints=cutpoints,
        glm=glm,
        hidden_size=hidden_size,
        num_hidden_layers=num_hidden_layers,
        dropout_rate=dropout_rate,
    )

    try:
        train(
            model=drn_model,
            criterion=lambda pred, y: drn_loss(
                pred,
                y,
                kl_alpha=kl_alpha,
                mean_alpha=mean_alpha,
                tv_alpha=0,
                dv_alpha=dv_alpha,
                kl_direction=kl_direction,
            ),
            train_dataset=TensorDataset(X_train, Y_train),
            val_dataset=TensorDataset(X_val, Y_val),
            batch_size=int(batch_size),
            epochs=2000,
            patience=patience,
            lr=lr,
            print_details=True,
            log_interval=30,
            device=device,
        )
    except Exception as e:
        print(f"Training failed: {e}")
        return 1e6

    if criteria == "CRPS":
        crps_score = compute_crps(drn_model, X_val, Y_val, Y_train.max().item(), device)
        return crps_score
    elif criteria == "NLL":
        with torch.no_grad():
            dists = drn_model.distributions(X_val)
            nll_score = -dists.log_prob(Y_val).mean().item()
            nll_score = nll_score if np.exp(-nll_score) > 0 else 1e10
        return nll_score
