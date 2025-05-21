import logging

import lightning as L  # type: ignore
import numpy as np
import torch
from drn import CANN, DDR, DRN, MDN, crps, ddr_cutpoints, drn_cutpoints
from lightning.pytorch.callbacks import EarlyStopping  # type: ignore
from torch.utils.data import DataLoader, TensorDataset

logging.getLogger("lightning.pytorch.utilities.rank_zero").addFilter(
    lambda record: "PU available: " not in record.getMessage()
)

TRAINER_OPTS = {
    "logger": False,
    "enable_checkpointing": False,
    "devices": 1,
    "deterministic": True,
    "enable_model_summary": False,
    "enable_progress_bar": False,
}

def compute_crps(
    model: torch.nn.Module,
    X_val: torch.Tensor,
    Y_val: torch.Tensor,
    y_train_max: float,
) -> float:
    """Compute the mean CRPS of model predictions on validation data."""
    grid_size = 3000
    grid = torch.linspace(0, y_train_max * 1.1, grid_size).unsqueeze(-1)

    with torch.no_grad():
        dists = model.distributions(X_val)
        cdfs = dists.cdf(grid)
        grid = grid.squeeze()
        crps_score = crps(Y_val, grid, cdfs).mean().item()

    return crps_score


def objective_cann(
    num_hidden_layers,
    hidden_size,
    dropout_rate,
    lr,
    batch_size,
    glm,
    X_train: torch.Tensor,
    Y_train: torch.Tensor,
    X_val: torch.Tensor,
    Y_val: torch.Tensor,
    accelerator: str,
    patience: int,
):
    train_loader = DataLoader(
        TensorDataset(X_train, Y_train), batch_size=int(batch_size), shuffle=True
    )
    val_loader = DataLoader(
        TensorDataset(X_val, Y_val), batch_size=int(batch_size), shuffle=False
    )

    cann = CANN(
        glm,
        num_hidden_layers=num_hidden_layers,
        hidden_size=hidden_size,
        dropout_rate=dropout_rate,
        learning_rate=lr,
    )

    try:
        trainer = L.Trainer(
            max_epochs=2000,
            accelerator=accelerator,
            callbacks=[EarlyStopping(monitor="val_loss", patience=patience)],
            **TRAINER_OPTS,
        )
        trainer.fit(cann, train_loader, val_loader)

        cann.update_dispersion(X_train, Y_train)
        if not torch.isfinite(cann.dispersion):
            raise ValueError("Dispersion is not finite")

    except Exception as e:
        print(f"Training failed: {e}")
        return 1e10, None

    crps_score = compute_crps(cann, X_val, Y_val, Y_train.max().item())
    return crps_score, cann


def objective_mdn(
    num_hidden_layers,
    hidden_size,
    dropout_rate,
    lr,
    num_components,
    batch_size,
    distribution,
    X_train: torch.Tensor,
    Y_train: torch.Tensor,
    X_val: torch.Tensor,
    Y_val: torch.Tensor,
    accelerator: str,
    patience: int,
):
    train_loader = DataLoader(
        TensorDataset(X_train, Y_train), batch_size=int(batch_size), shuffle=True
    )
    val_loader = DataLoader(
        TensorDataset(X_val, Y_val), batch_size=int(batch_size), shuffle=False
    )

    mdn = MDN(
        X_train.shape[1],
        num_components=num_components,
        hidden_size=hidden_size,
        num_hidden_layers=num_hidden_layers,
        dropout_rate=dropout_rate,
        distribution=distribution,
        learning_rate=lr,
    )

    try:
        trainer = L.Trainer(
            max_epochs=2000,
            accelerator=accelerator,
            callbacks=[EarlyStopping(monitor="val_loss", patience=patience)],
            **TRAINER_OPTS,
        )
        trainer.fit(mdn, train_loader, val_loader)

    except Exception as e:
        print(e)
        print(f"Training failed: {e}")
        return 1e10, None

    crps_score = compute_crps(mdn, X_val, Y_val, Y_train.max().item())

    return crps_score, mdn


def objective_ddr(
    num_hidden_layers,
    hidden_size,
    dropout_rate,
    lr,
    proportion,
    batch_size,
    X_train: torch.Tensor,
    Y_train: torch.Tensor,
    X_val: torch.Tensor,
    Y_val: torch.Tensor,
    accelerator: str,
    patience: int,
):
    train_loader = DataLoader(
        TensorDataset(X_train, Y_train), batch_size=int(batch_size), shuffle=True
    )
    val_loader = DataLoader(
        TensorDataset(X_val, Y_val), batch_size=int(batch_size), shuffle=False
    )

    cutpoints = ddr_cutpoints(
        c_0=max(Y_train.min().item() * 1.05, 0),
        c_K=Y_train.max().item() * 1.05,
        proportion=proportion,
        n=X_train.shape[0],
    )

    ddr = DDR(
        X_train.shape[1],
        cutpoints,
        num_hidden_layers=num_hidden_layers,
        hidden_size=hidden_size,
        dropout_rate=dropout_rate,
        learning_rate=lr,
    )

    try:
        trainer = L.Trainer(
            max_epochs=2000,
            accelerator=accelerator,
            callbacks=[EarlyStopping(monitor="val_loss", patience=patience)],
            **TRAINER_OPTS,
        )
        trainer.fit(ddr, train_loader, val_loader)
    except Exception as e:
        print(f"Training failed: {e}")
        return 1e10, ddr

    crps_score = compute_crps(ddr, X_val, Y_val, Y_train.max().item())
    return crps_score, ddr


def objective_drn(
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
    glm,
    kl_direction,
    criteria,
    X_train: torch.Tensor,
    Y_train: torch.Tensor,
    X_val: torch.Tensor,
    Y_val: torch.Tensor,
    accelerator: str,
    patience: int,
):
    train_loader = DataLoader(
        TensorDataset(X_train, Y_train), batch_size=int(batch_size), shuffle=True
    )
    val_loader = DataLoader(
        TensorDataset(X_val, Y_val), batch_size=int(batch_size), shuffle=False
    )

    cutpoints = drn_cutpoints(
        c_0=max(Y_train.min().item() * 1.05, 0),
        c_K=Y_train.max().item() * 1.05,
        y=Y_train.cpu().numpy(),
        proportion=proportion,
        min_obs=min_obs,
    )

    drn = DRN(
        num_features=X_train.shape[1],
        cutpoints=cutpoints,
        glm=glm,
        hidden_size=hidden_size,
        num_hidden_layers=num_hidden_layers,
        dropout_rate=dropout_rate,
        kl_alpha=kl_alpha,
        mean_alpha=mean_alpha,
        tv_alpha=0,
        dv_alpha=dv_alpha,
        kl_direction=kl_direction,
        learning_rate=lr,
    )

    try:
        trainer = L.Trainer(
            max_epochs=2000,
            accelerator=accelerator,
            callbacks=[EarlyStopping(monitor="val_loss", patience=patience)],
            **TRAINER_OPTS,
        )
        trainer.fit(drn, train_loader, val_loader)
    except Exception as e:
        print(f"Training failed: {e}")
        return 1e10, None

    if criteria == "CRPS":
        crps_score = compute_crps(drn, X_val, Y_val, Y_train.max().item())
        return crps_score, drn
    elif criteria == "NLL":
        with torch.no_grad():
            dists = drn.distributions(X_val)
            nll_score = -dists.log_prob(Y_val).mean().item()
            nll_score = nll_score if np.exp(-nll_score) > 0 else 1e10
        return nll_score, drn
    else:
        raise ValueError(f"Unknown criteria: {criteria}")
