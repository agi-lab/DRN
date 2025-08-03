from typing import Union
import torch
import numpy as np
import pandas as pd
from drn import CANN, DDR, DRN, MDN, crps


def compute_crps(model: torch.nn.Module, **args) -> float:
    """Compute the mean CRPS of model predictions on validation data."""
    X_val = args["X_val"]
    y_val = args["y_val"]
    y_train = args["y_train"]

    grid_size = 3000
    grid = torch.linspace(0, y_train.max().item() * 1.1, grid_size).unsqueeze(-1)

    with torch.no_grad():
        dists = model.distributions(X_val)
        cdfs = dists.cdf(grid)
        grid = grid.squeeze()
        return crps(y_val, grid, cdfs).mean().item()


def objective_cann(
    baseline,
    num_hidden_layers: int,
    hidden_size: int,
    dropout_rate: float,
    lr: float,
    **fit_kwargs,
) -> tuple[float, CANN | None]:
    """Objective for training and evaluating a CANN model."""
    fit_kwargs["batch_size"] = int(fit_kwargs["batch_size"])

    cann = CANN(
        baseline=baseline,
        num_hidden_layers=num_hidden_layers,
        hidden_size=hidden_size,
        dropout_rate=dropout_rate,
        learning_rate=lr,
    )

    try:
        cann.fit(**fit_kwargs)
        cann.eval()
    except Exception as e:
        print(f"Training failed: {e}")
        return 1e10, None

    score = compute_crps(cann, **fit_kwargs)
    return score, cann


def objective_mdn(
    num_hidden_layers: int,
    hidden_size: int,
    dropout_rate: float,
    lr: float,
    num_components: int,
    distribution: str,
    **fit_kwargs,
) -> tuple[float, MDN | None]:
    """Objective for training and evaluating an MDN model."""
    fit_kwargs["batch_size"] = int(fit_kwargs["batch_size"])

    mdn = MDN(
        num_components=num_components,
        hidden_size=hidden_size,
        num_hidden_layers=num_hidden_layers,
        dropout_rate=dropout_rate,
        distribution=distribution,
        learning_rate=lr,
    )

    try:
        mdn.fit(**fit_kwargs)
        mdn.eval()
    except Exception as e:
        print(f"Training failed: {e}")
        return 1e10, None

    score = compute_crps(mdn, **fit_kwargs)
    return score, mdn


def objective_ddr(
    num_hidden_layers: int,
    hidden_size: int,
    dropout_rate: float,
    lr: float,
    proportion: float,
    **fit_kwargs,
) -> tuple[float, DDR | None]:
    """Objective for training and evaluating a DDR model."""
    fit_kwargs["batch_size"] = int(fit_kwargs["batch_size"])

    ddr = DDR(
        num_hidden_layers=num_hidden_layers,
        hidden_size=hidden_size,
        dropout_rate=dropout_rate,
        learning_rate=lr,
        proportion=proportion,
    )

    try:
        ddr.fit(**fit_kwargs)
        ddr.eval()
    except Exception as e:
        print(f"Training failed: {e}")
        return 1e10, None

    score = compute_crps(ddr, **fit_kwargs)
    return score, ddr


def objective_drn(
    baseline,
    num_hidden_layers: int,
    hidden_size: int,
    dropout_rate: float,
    lr: float,
    kl_alpha: float,
    mean_alpha: float,
    dv_alpha: float,
    proportion: float,
    min_obs: int,
    kl_direction: str,
    criteria: str,
    **fit_kwargs,
) -> tuple[float, DRN | None]:
    """Objective for training and evaluating a DRN model."""
    fit_kwargs["batch_size"] = int(fit_kwargs["batch_size"])

    drn = DRN(
        baseline=baseline,
        hidden_size=hidden_size,
        num_hidden_layers=num_hidden_layers,
        dropout_rate=dropout_rate,
        kl_alpha=kl_alpha,
        mean_alpha=mean_alpha,
        tv_alpha=0,
        dv_alpha=dv_alpha,
        kl_direction=kl_direction,
        learning_rate=lr,
        proportion=proportion,
        min_obs=min_obs,
    )

    try:
        drn.fit(**fit_kwargs)
        drn.eval()
    except Exception as e:
        print(f"Training failed: {e}")
        return 1e10, None

    if criteria == "CRPS":
        score = compute_crps(drn, **fit_kwargs)

    elif criteria == "NLL":
        X_val = fit_kwargs["X_val"]
        y_val = fit_kwargs["y_val"]
        with torch.no_grad():
            dists = drn.distributions(X_val)
            nll = -dists.log_prob(y_val).mean().item()
            score = nll if np.exp(-nll) > 0 else 1e10
    else:
        raise ValueError(f"Unknown criteria: {criteria}")

    return score, drn
