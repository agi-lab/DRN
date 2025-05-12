import numpy as np
import torch
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


# Objective function to minimize for CANN
def objective_cann(
    params,
    X_train,
    Y_train,
    X_val,
    Y_val,
    train_dataset,
    val_dataset,
    glm,
    distribution,
    patience=50,
):
    num_hidden_layers, hidden_size, dropout_rate, lr, batch_size = params

    num_hidden_layers = int(num_hidden_layers)
    hidden_size = int(hidden_size)
    batch_size = int(batch_size)

    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")

    torch.manual_seed(23)
    cann = CANN(
        glm,
        num_hidden_layers=num_hidden_layers,
        hidden_size=hidden_size,
        dropout_rate=dropout_rate,
    )

    try:
        torch.manual_seed(23)
        train(
            cann,
            (
                gaussian_deviance_loss
                if distribution == "gaussian"
                else gamma_deviance_loss
            ),
            train_dataset,
            val_dataset,
            epochs=2000,
            lr=lr,
            patience=patience,
            batch_size=batch_size,
        )
        cann.update_dispersion(X_train, Y_train)
        if not torch.isfinite(cann.dispersion):
            raise ValueError("Dispersion is not finite")
        
    except Exception as e:
        print(f"Training failed: {e}")
        return 1e6

    grid_size = 3000
    grid = (
        torch.linspace(0, np.max(Y_train.detach().numpy()) * 1.1, grid_size)
        .unsqueeze(-1)
        .to(device)
    )

    cann.eval()
    with torch.no_grad():
        dists = cann.distributions(X_val)
        cdfs = dists.cdf(grid)
        grid = grid.squeeze()
        crps_score = crps(Y_val, grid, cdfs).mean().item()

    return crps_score


# Objective function to minimise for MDN
def objective_mdn(
    params,
    X_train,
    Y_train,
    X_val,
    Y_val,
    train_dataset,
    val_dataset,
    distribution,
    patience=50,
):
    num_hidden_layers, hidden_size, dropout_rate, lr, num_components, batch_size = (
        params
    )

    num_hidden_layers = int(num_hidden_layers)
    hidden_size = int(hidden_size)
    batch_size = int(batch_size)
    num_components = int(num_components)

    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")

    torch.manual_seed(23)
    mdn = MDN(
        X_train.shape[1],
        num_components=num_components,
        hidden_size=hidden_size,
        num_hidden_layers=num_hidden_layers,
        dropout_rate=dropout_rate,
        distribution=distribution,
    )

    try:
        torch.manual_seed(23)
        train(
            mdn,
            gaussian_mdn_loss if distribution == "gaussian" else gamma_mdn_loss,
            train_dataset,
            val_dataset,
            lr=lr,
            batch_size=batch_size,
            epochs=2000,
            patience=patience,
            device=device,
        )
        mdn.eval()

    except Exception as e:
        print(e)
        print(f"Training failed: {e}")
        return 1e6

    grid_size = 3000
    grid = (
        torch.linspace(0, np.max(Y_train.detach().numpy()) * 1.1, grid_size)
        .unsqueeze(-1)
        .to(device)
    )

    mdn.eval()
    with torch.no_grad():
        dists = mdn.distributions(X_val)
        cdfs = dists.cdf(grid)
        grid = grid.squeeze()
        crps_score = crps(Y_val, grid, cdfs).mean().item()

    return crps_score


# Objective function to minimise for DDR
def objective_ddr(
    params, X_train, Y_train, X_val, Y_val, train_dataset, val_dataset, patience=30
):
    num_hidden_layers, hidden_size, dropout_rate, lr, proportion, batch_size = params

    num_hidden_layers = int(num_hidden_layers)
    hidden_size = int(hidden_size)
    batch_size = int(batch_size)

    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")

    cutpoints_DDR = ddr_cutpoints(
        c_0=(
            np.min(Y_train.detach().numpy()) * 1.05
            if np.min(Y_train.detach().numpy()) < 0
            else 0.0
        ),
        c_K=np.max(Y_train.detach().numpy()) * 1.05,
        y=Y_train.detach().numpy(),
        p=proportion,
    )

    torch.manual_seed(23)
    ddr_model = DDR(
        X_train.shape[1],
        cutpoints_DDR,
        num_hidden_layers=num_hidden_layers,
        hidden_size=hidden_size,
        dropout_rate=dropout_rate,
    )

    try:
        torch.manual_seed(23)
        train(
            ddr_model,
            ddr_loss,
            train_dataset,
            val_dataset,
            epochs=2000,
            lr=lr,
            patience=patience,
            batch_size=batch_size,
        )
    except Exception as e:
        print(f"Training failed: {e}")
        return 1e6

    grid_size = 3000
    grid = (
        torch.linspace(0, np.max(Y_train.detach().numpy()) * 1.1, grid_size)
        .unsqueeze(-1)
        .to(device)
    )

    ddr_model.eval()
    with torch.no_grad():
        dists = ddr_model.distributions(X_val)
        cdfs = dists.cdf(grid)
        grid = grid.squeeze()
        crps_score = crps(Y_val, grid, cdfs).mean().item()

    return crps_score


# Objective function to minimise for DRN
def objective_drn(
    params,
    X_train,
    Y_train,
    X_val,
    Y_val,
    train_dataset,
    val_dataset,
    glm,
    kl_direction="forwards",
    criteria="CRPS",
    patience=30,
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

    # Since those integer values are numpy.int64, and that breaks some things, manually convert to Python ints
    num_hidden_layers = int(num_hidden_layers)
    hidden_size = int(hidden_size)
    batch_size = int(batch_size)
    min_obs = int(min_obs)

    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")

    cutpoints = drn_cutpoints(
        c_0=(
            np.min(Y_train.detach().numpy()) * 1.05
            if np.min(Y_train.detach().numpy()) < 0
            else 0.0
        ),
        c_K=np.max(Y_train.detach().numpy()) * 1.05,
        p=proportion,
        y=Y_train.detach().numpy(),
        min_obs=min_obs,
    )
    torch.manual_seed(23)
    drn_model = DRN(
        num_features=X_train.shape[1],
        cutpoints=cutpoints,
        glm=glm,
        hidden_size=hidden_size,
        num_hidden_layers=num_hidden_layers,
        dropout_rate=dropout_rate,
    )

    # Train the model with the provided hyperparameters
    try:
        torch.manual_seed(23)
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
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            batch_size=batch_size,
            epochs=2000,
            patience=patience,
            lr=lr,
            print_details=True,
            log_interval=30,
            device=device,
        )
    except Exception as e:
        print(f"Training failed: {e}")
        return 1e6  # Return a large number in case of failure

    # Calculate validation loss and return it
    grid_size = 3000  # Increase this to get more accurate CRPS estimates
    grid = (
        torch.linspace(0, np.max(Y_train.detach().numpy()) * 1.1, grid_size)
        .unsqueeze(-1)
        .to(device)
    )

    drn_model.eval()
    with torch.no_grad():
        dists = drn_model.distributions(X_val)
        cdfs = dists.cdf(grid)
        grid = grid.squeeze()
        crps_score = crps(Y_val, grid, cdfs).mean().item()
        nll_score = -dists.log_prob(Y_val).mean().item()
        nll_score = nll_score if np.exp(-nll_score) > 0 else 1e10

    print(f"CRPS: {crps_score}, NLL: {nll_score}")

    return crps_score if criteria == "CRPS" else nll_score
