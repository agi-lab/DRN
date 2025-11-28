from typing import Callable, Sequence, Mapping

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
import statsmodels.api as sm
import torch
from drn import crps, rmse
from scipy.stats import wilcoxon
from tqdm.auto import trange


# Quantile Residuals and Calibration
def quantile_residuals(y, F_, interval):
    if y < interval[0]:
        return 0
    if y > interval[len(interval) - 1]:
        return 1
    for i in range(len(interval) - 1):
        if y > interval[i] and y <= interval[i + 1]:
            idx_low = i
            idx_up = i + 1
            return 0.5 * (F_[idx_low] + F_[idx_up])


def quantile_points(cdfs, response, grid, model_names=None):
    if model_names is None:
        model_names = list(cdfs.keys())  # fallback if not provided

    response = np.array(response)
    all_points = {name: [[0]] * len(response) for name in model_names}

    for k in trange(len(response)):
        for name in model_names:
            all_points[name][k] = quantile_residuals(
                response[k], cdfs[name][:, k].detach().numpy(), grid.detach().numpy()
            )

    return all_points


def quantile_residuals_plots(model_points):
    model_names = list(model_points.keys())

    # Compute quantile residuals for each model
    quantiles = [
        np.array(scipy.stats.norm.ppf(model_points[name]))
        for name in model_names
    ]

    n_models = len(model_names)
    n_cols = 2 if n_models > 1 else 1
    n_rows = int(np.ceil(n_models / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(26, 26))
    axes = np.atleast_1d(axes).ravel()   # always a flat array

    # Plot only the axes we need
    for i, (name, q) in enumerate(zip(model_names, quantiles)):
        sm.qqplot(q, line="45", ax=axes[i])
        axes[i].set_title(name, fontsize=45, color="black")
        axes[i].set_ylim(-5, 5)
        axes[i].set_xlim(-5, 5)

    # Remove any unused axes so no empty grids
    for j in range(n_models, len(axes)):
        fig.delaxes(axes[j])

    # Label + ticks only on used axes
    for ax in axes[:n_models]:
        ax.set_xlabel("Theoretical Quantiles", fontsize=45)
        ax.set_ylabel("Sample Quantiles", fontsize=45)
        ax.tick_params(axis="both", which="major", labelsize=40)

    fig.suptitle("Quantile Residuals", fontsize=60, y=0.99)
    plt.tight_layout(pad=2)

    return fig, axes[:n_models]



# Find the index (from the list lst_new) that gives the closest value to the given scalar y
def closest_index(y, lst_new):
    low, high = 0, len(lst_new) - 1
    while low < high - 1:
        mid = (low + high) // 2
        if lst_new[mid] == y:
            return mid
        elif lst_new[mid] < y:
            low = mid
        else:
            high = mid

    if abs(lst_new[low] - y) <= abs(lst_new[high] - y):
        return low
    else:
        return high


def calibration_plot_stats(cdfs_, grid, responses):
    Q_predicted = [[0]] * len(responses)
    Q_empirical = [[0]] * len(responses)

    cdfs_ = cdfs_.T

    for k in trange(len(responses)):
        y = responses[k]
        Q_predicted[k] = np.array(cdfs_[k])[closest_index(y, grid)].item()

    sorted_indices = np.argsort(Q_predicted)
    sorted_F_y_given_x = np.array(Q_predicted)[sorted_indices]
    empirical_probs = (np.arange(1, len(responses) + 1) - 0.5) / len(responses)

    print(sorted_F_y_given_x.shape, empirical_probs.shape)
    return (sorted_F_y_given_x, empirical_probs)


def calibration_plot(cdfs_, y, grid, model_names=None):
    if model_names is None:
        model_names = list(cdfs_.keys())

    responses = np.array(y)

    cmap = plt.get_cmap("tab10")
    colors = [cmap(i % cmap.N) for i in range(len(model_names))]
    predictions = []

    for model in model_names:
        Q_pred, Q_emp = calibration_plot_stats(
            cdfs_[model].detach().numpy(),
            grid.detach().numpy(),
            responses,
        )
        stats = np.sum((np.array(Q_pred) - np.array(Q_emp)) ** 2) / len(responses)
        predictions.append((model, Q_pred, Q_emp, stats))

    n_models = len(predictions)
    n_cols = 2 if n_models > 1 else 1
    n_rows = int(np.ceil(n_models / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 16))
    axes = np.atleast_1d(axes).ravel()

    for i, (model, Q_pred, Q_emp, stats) in enumerate(predictions):
        axes[i].scatter(
            Q_pred,
            Q_emp,
            s=14,
            color=colors[i],
            label=f"{model} \n $\\sum_j (p_j-\\hat p_j)^2/n= {round(stats*len(responses), 4)}$",
        )
        axes[i].plot([0, 1], [0, 1], ls="--", color="red")
        axes[i].set_xlabel("Predicted: $\\hat{p}$", fontsize=30)
        axes[i].set_ylabel("Empirical: $p$", fontsize=30)
        axes[i].set_title(f"Calibration Plot: {model}", fontsize=36)

        legend = axes[i].legend(prop={"size": 22}, scatterpoints=1)
        for handle in legend.legend_handles:
            handle.set_sizes([40])

    # Remove any unused axes so they don't show as empty panels
    for j in range(n_models, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    return fig, axes[:n_models]



# Wilcoxon Test


def print_wilcoxon_test(
    glm_metrics, cann_metrics, mdn_metrics, ddr_metrics, drn_metrics
):
    # Perform the Wilcoxon Signed-Rank Test
    stat, p_value = wilcoxon(drn_metrics, glm_metrics, alternative="less")
    print("DRN < GLM")
    print("Wilcoxon Signed-Rank Test statistic:", stat)
    print("P-value:", p_value)

    stat, p_value = wilcoxon(drn_metrics, cann_metrics, alternative="less")
    print("DRN < CANN")
    print("Wilcoxon Signed-Rank Test statistic:", stat)
    print("P-value:", p_value)

    stat, p_value = wilcoxon(drn_metrics, mdn_metrics, alternative="less")
    print("DRN < MDN")
    print("Wilcoxon Signed-Rank Test statistic:", stat)
    print("P-value:", p_value)

    stat, p_value = wilcoxon(drn_metrics, ddr_metrics, alternative="less")
    print("DRN < DDR")
    print("Wilcoxon Signed-Rank Test statistic:", stat)
    print("P-value:", p_value)


def nll_wilcoxon_test(dists, Y_target, dataset="Test"):
    # NLL data
    nll_model_glm = -dists["GLM"].log_prob(Y_target).squeeze().detach().numpy()
    nll_model_cann = -dists["CANN"].log_prob(Y_target).squeeze().detach().numpy()
    nll_model_mdn = -dists["MDN"].log_prob(Y_target).squeeze().detach().numpy()
    nll_model_ddr = -dists["DDR"].log_prob(Y_target).squeeze().detach().numpy()
    nll_model_drn = -dists["DRN"].log_prob(Y_target).squeeze().detach().numpy()

    print("--------------------------------------------")
    print(f"{dataset} Data")
    print("--------------------------------------------")

    print_wilcoxon_test(
        nll_model_glm, nll_model_cann, nll_model_mdn, nll_model_ddr, nll_model_drn
    )


def crps_wilcoxon_test(cdfs_, Y_target, grid, dataset="Test"):
    # CRPS data
    crps_model_drn = crps(Y_target, grid, cdfs_["DRN"]).squeeze().detach().numpy()
    crps_model_glm = crps(Y_target, grid, cdfs_["GLM"]).squeeze().detach().numpy()
    crps_model_cann = crps(Y_target, grid, cdfs_["CANN"]).squeeze().detach().numpy()
    crps_model_mdn = crps(Y_target, grid, cdfs_["MDN"]).squeeze().detach().numpy()
    crps_model_ddr = crps(Y_target, grid, cdfs_["DDR"]).squeeze().detach().numpy()

    print("--------------------------------------------")
    print(f"{dataset} Data")
    print("--------------------------------------------")

    print_wilcoxon_test(
        crps_model_glm, crps_model_cann, crps_model_mdn, crps_model_ddr, crps_model_drn
    )


def rmse_wilcoxon_test(dists_, Y_target, dataset="Test"):
    # MSE data
    se_drn = (
        dists_["DRN"].mean.squeeze().detach().numpy()
        - Y_target.squeeze().detach().numpy()
    ) ** 2
    se_glm = (
        dists_["GLM"].mean.squeeze().detach().numpy()
        - Y_target.squeeze().detach().numpy()
    ) ** 2
    se_cann = (
        dists_["CANN"].mean.squeeze().detach().numpy()
        - Y_target.squeeze().detach().numpy()
    ) ** 2
    se_mdn = (
        dists_["MDN"].mean.squeeze().detach().numpy()
        - Y_target.squeeze().detach().numpy()
    ) ** 2
    se_ddr = (
        dists_["DDR"].mean.squeeze().detach().numpy()
        - Y_target.squeeze().detach().numpy()
    ) ** 2

    print("--------------------------------------------")
    print(f"{dataset} Data")
    print("--------------------------------------------")

    print_wilcoxon_test(se_glm, se_cann, se_mdn, se_ddr, se_drn)


def quantile_score(y_true, y_pred, p):
    """
    Compute the quantile score for predictions at a specific quantile.

    :param y_true: Actual target values as a Pandas Series or PyTorch tensor.
    :param y_pred: Predicted target values as a numpy array or PyTorch tensor.
    :param p: The cumulative probability as a float
    :return: The quantile score as a PyTorch tensor.
    """
    # Ensure that y_true and y_pred are PyTorch tensors
    y_true = (
        torch.Tensor(y_true.values) if not isinstance(y_true, torch.Tensor) else y_true
    )
    y_pred = torch.Tensor(y_pred) if not isinstance(y_pred, torch.Tensor) else y_pred
    # Reshape y_pred to match y_true if necessary and compute the error
    e = y_true - y_pred.reshape(y_true.shape)
    # Compute the quantile score
    return torch.where(y_true >= y_pred, p * e, (1 - p) * -e).mean()



def quantile_losses(
    p,
    model,
    model_name,
    X,
    y,
    max_iter=1000,
    tolerance=5e-5,
    l=None,
    u=None,
    print_score=False,
    mean_tensor=True,
):
    """
    Calculate and optionally print the quantile loss for the given data and model.

    Args:
        p (float): Target cumulative probability in (0,1), e.g. 0.9 for the 90th percentile.
        model: Trained model.
        model_name (str): A label whose PREFIX indicates the model family:
            - startswith("GLM")  -> call model.quantiles(...)
            - startswith("MDN")  -> call model.quantiles(...)
            - startswith("DRN")  -> call model.predict(X).quantiles(...)
          (Prefix check is case-insensitive.)
        X: Features (Pandas DataFrame or numpy array).
        y: True targets (Pandas Series or numpy array).
        max_iter (int): Max iterations for quantile search.
        tolerance (float): Convergence tolerance for quantile search.
        l, u (float|None): Optional lower/upper bounds for search.
        print_score (bool): Whether to print the score.

    Returns:
        torch.Tensor: The quantile loss.
    """
    fam = model_name.strip().upper()

    if fam.startswith("GLM") or fam.startswith("MDN") or fam.startswith("CANN"):
        predicted_quantiles = model.quantiles(
            X, [p * 100], max_iter=max_iter, tolerance=tolerance, l=l, u=u
        )
    elif fam.startswith("DRN") or fam.startswith("DDR"):
        if fam.startswith("DRN"):
            distribution = model.glm.distribution
            l = None if distribution in ['gaussian'] else 0
        predicted_quantiles = model.predict(X).quantiles(
            [p * 100], max_iter=max_iter, tolerance=tolerance, l=l, u=u
        )
    else:
        raise ValueError(
            f"Unknown model_name prefix for quantiles: '{model_name}'. "
            "Use a name starting with GLM, MDN, or DRN (e.g., 'GLM_gamma', 'MDN_plain', 'DRN_std')."
        )
    import drn
    score = drn.quantile_score(y, predicted_quantiles, p, mean_tensor)

    if print_score:
        print(f"{model_name}: {score:.5f}")

    return score

def ql90_wilcoxon_test(models, X_features, Y_target, y_train, dataset="Test"):
    glm, cann, mdn, ddr, drn = models

    # 90% QL data
    ql_glm = (
        quantile_losses(
            0.9,
            glm,
            "GLM",
            X_features,
            Y_target,
            max_iter=1000,
            tolerance=1e-4,
            l=torch.Tensor([0]),
            u=torch.Tensor([np.max(y_train) + 3 * (np.max(y_train) - np.min(y_train))]),
        )
        .squeeze()
        .detach()
        .numpy()
    )
    ql_cann = (
        quantile_losses(
            0.9,
            cann,
            "CANN",
            X_features,
            Y_target,
            max_iter=1000,
            tolerance=1e-4,
            l=torch.Tensor([0]),
            u=torch.Tensor([np.max(y_train) + 3 * (np.max(y_train) - np.min(y_train))]),
        )
        .squeeze()
        .detach()
        .numpy()
    )
    ql_mdn = (
        quantile_losses(
            0.9,
            mdn,
            "MDN",
            X_features,
            Y_target,
            max_iter=1000,
            tolerance=1e-4,
            l=torch.Tensor([0]),
            u=torch.Tensor([np.max(y_train) + 3 * (np.max(y_train) - np.min(y_train))]),
        )
        .squeeze()
        .detach()
        .numpy()
    )
    ql_ddr = (
        quantile_losses(
            0.9,
            ddr,
            "DDR",
            X_features,
            Y_target,
            max_iter=1000,
            tolerance=1e-4,
            l=torch.Tensor([0]),
            u=torch.Tensor([np.max(y_train) + 3 * (np.max(y_train) - np.min(y_train))]),
        )
        .squeeze()
        .detach()
        .numpy()
    )
    ql_drn = (
        quantile_losses(
            0.9,
            drn,
            "DRN",
            X_features,
            Y_target,
            max_iter=1000,
            tolerance=1e-4,
            l=torch.Tensor([0]),
            u=torch.Tensor([np.max(y_train) + 3 * (np.max(y_train) - np.min(y_train))]),
        )
        .squeeze()
        .detach()
        .numpy()
    )

    print("--------------------------------------------")
    print(f"{dataset} Data")
    print("--------------------------------------------")

    print_wilcoxon_test(ql_glm, ql_cann, ql_mdn, ql_ddr, ql_drn)


def generate_latex_table(
    nlls_val,
    crps_val,
    rmse_val,
    ql_90_val,
    nlls_test,
    crps_test,
    rmse_test,
    ql_90_test,
    model_names,
    label_txt="Evaluation Metrics Table",
    caption_txt="Evaluation Metrics Table.",
    scaling_factor=1.0,
):
    header_row = (
        "\\begin{center}\n"
        + "\captionof{table}{"
        + f"{caption_txt}"
        + "}\n"
        + "\label{"
        + f"{label_txt}"
        + "}\n"
        + "\scalebox{"
        + f"{scaling_factor}"
        + "}{\n"
        + "\\begin{tabular}{l|cccc|cccc}\n\\toprule\n\\toprule\n"
        + "&  \multicolumn{4}{c}{$\mathcal{D}_{\\text{Validation}}$}"
        + "& \multicolumn{4}{c}{ $\mathcal{D}_{\\text{Test}}$}\\\\ \n"
        + " \cmidrule{2-5}  \cmidrule{6-9} $\\text{Model}$ $\\backslash$ $\\text{Metrics}$"
        + " & NLL & CRPS & RMSE & 90\% QL & NLL & CRPS & RMSE & 90\% QL \\\\ \\midrule"
    )
    rows = [header_row]

    for name in model_names:
        row = (
            f"{name} &  {(nlls_val[name].mean()):.4f}"
            f" &  {(crps_val[name].mean()):.4f} "
            f" & {(rmse_val[name].mean()):.4f} "
            f" & {(ql_90_val[name].mean()):.4f} "
            f" & {(nlls_test[name].mean()):.4f} "
            f" & {(crps_test[name].mean()):.4f} "
            f" & {(rmse_test[name].mean()):.4f} "
            f" & {(ql_90_test[name].mean()):.4f} \\\\ "
        )
        rows.append(row)

    table = (
        "\n".join(rows)
        + "\n\\bottomrule\n\\bottomrule"
        + "\n\\end{tabular}"
        + "\n}"
        + "\n\end{center}"
    )
    return table


def calculate_metrics(
    models, names, X_test_data, Y_test_data, y_train, train_size, seed_index, tidy_df_output = True, grid_size = 3000, vec_output = False
):
    """
    Compute NLL, CRPS, RMSE, and QL(0.9) for each model, and return a tidy DataFrame
    with columns ['train_size', 'seed_index', 'model', 'metric', 'value'].

    This version builds the DataFrame rows on-the-fly and uses a loop to avoid
    repeating `rows.append` for each metric.
    """
    # 1) Precompute a common grid for CRPS
    GRID_SIZE = grid_size
    max_y = float(np.max(y_train)) * 1.1
    grid = torch.linspace(0.0001, max_y, GRID_SIZE).unsqueeze(-1)  # (GRID_SIZE, 1)
    grid_flat = grid.squeeze()  # (GRID_SIZE,)

    rows = []

    for model, model_name in zip(models, names):
        # 2) Get predictive distribution on test set
        dist = model.predict(X_test_data)

        # 3) Compute CDF over grid for CRPS
        cdf_vals = dist.cdf(grid)  # (N_test, GRID_SIZE)

        # 4) Negative Log‐Likelihood
        nll_val = -dist.log_prob(Y_test_data).mean().item()

        # 5) CRPS
        crps_val = crps(Y_test_data, grid_flat, cdf_vals).mean().item()

        # 6) RMSE
        rmse_val = rmse(Y_test_data.detach(), dist.mean).item()

        # # 7) Quantile Loss at α = 0.9
        # lower_bound = torch.Tensor([0.0])
        # upper_bound = torch.Tensor(
        #     [np.max(y_train) + 3 * (np.max(y_train) - np.min(y_train))]
        # )
        ql90_val = quantile_losses(
            0.9,
            model,
            model_name,
            X_test_data,
            Y_test_data,
            max_iter=1000,
            tolerance=1e-4,
            # l=lower_bound,
            # u=upper_bound,
        ).item()

        # 8) Collect metric names and values in a list, then loop to append rows
        metric_items = [
            ("NLL", nll_val),
            ("CRPS", crps_val),
            ("RMSE", rmse_val),
            ("QL90", ql90_val),
        ]
        if tidy_df_output:
            for metric_name, metric_value in metric_items:
                rows.append(
                    {
                        "train_size": train_size,
                        "seed_index": seed_index,
                        "model": model_name,
                        "metric": metric_name,
                        "value": metric_value,
                    }
                )
        else:
            rows.append({
                "model": model_name,
                "mean_crps": crps_val,
                "mean_nll": nll_val,
                "ql_90": ql90_val,
                "rmse": rmse_val,
            })
    if tidy_df_output:
        tidy_df = pd.DataFrame(
            rows, columns=["train_size", "seed_index", "model", "metric", "value"]
        )
        return tidy_df
    else:
        return pd.DataFrame(rows)


def calculate_metrics_vec(
    models, names, X_test_data, Y_test_data, y_train, train_size, seed_index, tidy_df_output = True, grid_size = 3000
):
    """
    Compute NLL, CRPS, RMSE, and QL(0.9) for each model, and return a tidy DataFrame
    with columns ['train_size', 'seed_index', 'model', 'metric', 'value'].

    This version builds the DataFrame rows on-the-fly and uses a loop to avoid
    repeating `rows.append` for each metric.
    """
    # 1) Precompute a common grid for CRPS
    GRID_SIZE = grid_size
    max_y = float(np.max(y_train)) * 1.1
    grid = torch.linspace(0.0001, max_y, GRID_SIZE).unsqueeze(-1)  # (GRID_SIZE, 1)
    grid_flat = grid.squeeze()  # (GRID_SIZE,)

    rows = []

    for model, model_name in zip(models, names):
        # 2) Get predictive distribution on test set
        dist = model.predict(X_test_data)

        # 3) Compute CDF over grid for CRPS
        cdf_vals = dist.cdf(grid)  # (N_test, GRID_SIZE)

        # 4) Negative Log‐Likelihood
        nll_vec_cpu = -dist.log_prob(Y_test_data)

        # 5) CRPS
        crps_vec_cpu = crps(Y_test_data, grid_flat, cdf_vals)

        # 6) RMSE
        se_vec_cpu = (Y_test_data.detach()-dist.mean)**2
        
        # 7) Quantile Loss at α = 0.9
        ql_vec_cpu = quantile_losses(
            0.9,
            model,
            model_name,
            X_test_data,
            Y_test_data,
            max_iter=1000,
            tolerance=1e-4,
            mean_tensor=False,
            print_score=False,
        )

        if tidy_df_output:
            # push full tensors in the 'value' column
            rows.extend([
                {
                    "train_size": train_size,
                    "seed_index": seed_index,
                    "model": model_name,
                    "metric": "NLL",
                    "value": nll_vec_cpu,   # full tensor
                },
                {
                    "train_size": train_size,
                    "seed_index": seed_index,
                    "model": model_name,
                    "metric": "CRPS",
                    "value": crps_vec_cpu,  # full tensor
                },
                {
                    "train_size": train_size,
                    "seed_index": seed_index,
                    "model": model_name,
                    "metric": "RMSE",       # per-sample squared-error contributions
                    "value": se_vec_cpu,    # full tensor
                },
                {
                    "train_size": train_size,
                    "seed_index": seed_index,
                    "model": model_name,
                    "metric": f"QL{int(round(0.9*100))}",
                    "value": ql_vec_cpu,    # full tensor
                },
            ])
        else:
            # Non-tidy fallback: keep shape and names explicit (object columns)
            rows.append({
                "model": model_name,
                "nll": nll_vec_cpu,
                "crps": crps_vec_cpu,
                "rmse_contrib": se_vec_cpu,
                f"ql_{int(round(0.9*100))}": ql_vec_cpu,
            })

    if tidy_df_output:
        tidy_df = pd.DataFrame(rows, columns=["train_size", "seed_index", "model", "metric", "value"])
        return tidy_df
    else:
        return pd.DataFrame(rows)


def evaluate_models_metrics(models, names, X_test_df, Y_test_tensor, grid=None):
    """
    Compute CRPS, NLL, QL90, RMSE for each model.
    - CRPS uses drn.crps with predicted CDFs on a common grid.
    - NLL approximated via finite-difference PDF from CDF and interpolation at y.
    - QL90 computed by inverting the CDF at tau=0.9.
    - RMSE computed against the CDF-implied predictive mean.
    """
    import drn
    # Default grid based on Y_test span
    if grid is None:
        y_np0 = Y_test_tensor.detach().cpu().numpy()
        c0 = 0.0 if y_np0.min() >= 0 else float(y_np0.min() * 1.1)
        cK = float(y_np0.max() * 1.1)
        grid = torch.linspace(c0, cK, steps=1024).view(-1, 1)

    grid_1d_torch = grid.squeeze()
    grid_np = grid_1d_torch.detach().cpu().numpy()

    X_test_tensor = torch.tensor(X_test_df.values, dtype=torch.float32)
    rows = []

    for model, name in zip(models, names):
        preds = model.predict(X_test_tensor)
        cdfs_t = preds.cdf(grid)  # expect shape (N, G); torch tensor or numpy
        

        # Ensure torch tensor for CRPS
        if not torch.is_tensor(cdfs_t):
            cdfs_torch = torch.tensor(cdfs_t, dtype=torch.float32)
        else:
            cdfs_torch = cdfs_t

        # --- CRPS ---
        crps = drn.crps(Y_test_tensor, grid_1d_torch, cdfs_torch).mean()
        
        # --- NLL (via PDF from CDF + interpolation at each y) ---
        nll = -model.predict(X_test_tensor).log_prob(Y_test_tensor).mean()

        # # --- QL90 (pinball loss at tau=0.9) ---
        ql90 = quantile_losses(
                0.9,
                model,
                name,
                X_test_tensor,
                Y_test_tensor,
                max_iter=1000,
                tolerance=1e-4,
            )
        # --- RMSE (using CDF-implied predictive mean) ---
        Y_hat = model.predict(X_test_tensor).mean
        rmse = drn.rmse(Y_test_tensor, Y_hat)

        rows.append({
            "model": name,
            "mean_crps": crps.detach().numpy(),
            "mean_nll": nll.detach().numpy(),
            "ql_90": ql90.detach().numpy(),
            "rmse": rmse.detach().numpy(),
        })

    return pd.DataFrame(rows)

def process_data_with_std(data_dict, remove_outliers=False, z_thresh=2.0):
    x = sorted(map(int, data_dict.keys()))
    y, y_std, y_min, y_max = [], [], [], []

    for k in x:
        values = np.array(data_dict[k])

        if remove_outliers:
            mean = np.mean(values)
            std = np.std(values)
            z_scores = (values - mean) / std
            values = values[np.abs(z_scores) <= z_thresh]

        mean = np.mean(values)
        std = np.std(values)

        y.append(mean)
        y_std.append(std)
        y_min.append(mean - std)
        y_max.append(mean + std)

    return x, y, y_min, y_max


# Function to compute mean and standard deviation
def compute_mean_std(data_dict, keys_to_extract=[1000, 3000, 6000]):
    mean_std_dict = {}
    for model, values in data_dict.items():
        mean_std_dict[model] = {}
        for key in keys_to_extract:
            data = np.array(values[key])
            mean_std_dict[model][key] = (np.mean(data), np.std(data))
    return mean_std_dict


def plot_metrics_grid(df: pd.DataFrame):
    fig, axes = plt.subplots(2, 2, figsize=(18, 14))
    axes = axes.flatten()

    sizes = df["train_size"].unique()
    metric_names = list(df["metric"].unique())
    model_names = list(df["model"].unique())

    # A fixed color palette (as before)
    colors = ["red", "orange", "blue", "black"]

    for i, metric in enumerate(metric_names):
        ax = axes[i]

        for model_name, color in zip(model_names, colors):
            # 1) Filter df for this (metric, model)
            subset = df[(df["metric"] == metric) & (df["model"] == model_name)]

            # 2) Group by size, collect a list of all “value” entries for each size
            #    This yields a Series indexed by size, whose values are LISTS of floats.
            grouped: pd.Series = subset.groupby("train_size")["value"].apply(list)

            # 3) Convert that into a plain dict: { size: [val1, val2, ...], ... }
            data_subset: dict[int, list[float]] = grouped.to_dict()

            # 4) Pass that dict into process_fn to get (x, y, y_min, y_max)
            x_vals, y_mean, y_min, y_max = process_data_with_std(data_subset)

            # 5) Plot the central line + shaded band, exactly as before
            ax.plot(x_vals, y_mean, color=color, label=model_name, linewidth=2)
            ax.fill_between(x_vals, y_min, y_max, color=color, alpha=0.1)

        # Formatting (same as your original)
        ax.set_title(
            f"{metric} Comparison Across Different Baseline Models", fontsize=22
        )
        ax.set_xlabel("Training Size", fontsize=18)
        ax.set_ylabel(metric, fontsize=18)
        ax.legend(fontsize=16)
        ax.grid(True, linestyle="--", alpha=0.7)

        ax.set_xticks(sizes)
        # If you want to relabel them as [600, 1800, 3600], do so here:
        ax.set_xticklabels([600, 1800, 3600], fontsize=16)

        ax.tick_params(axis="y", labelsize=16, width=2)
        ax.tick_params(axis="x", labelsize=16, width=2)

    plt.tight_layout()


def rank_models_per_seed(tidy_df: pd.DataFrame, metric_name: str):
    """
    Given a long/tidy DataFrame with columns
      ['train_size', 'seed_index', 'model', 'metric', 'value'],
    filter to (metric==metric_name), pivot to
    (index=seed_index, columns=model, values=value), then rank each row
    (ascending=True since lower is better). Returns a DataFrame of ranks
    (shape: n_seeds × n_models).
    """
    df_sub = tidy_df[(tidy_df["metric"] == metric_name)].copy()

    # Pivot so that each row is one seed_index, columns are model names
    pivot = df_sub.pivot(index="seed_index", columns="model", values="value")

    # Rank each row (axis=1) — “method='min'” and ascending=True means smaller value → rank 1
    ranks = pivot.rank(axis=1, method="min", ascending=True).astype(int)

    print(f"\n===== Ranks for {metric_name} =====")
    display(ranks)
    return ranks


def generate_latex_table_all(
    nlls_train,
    crps_train,
    rmse_train,
    ql_90_train,
    nlls_val,
    crps_val,
    rmse_val,
    ql_90_val,
    nlls_test,
    crps_test,
    rmse_test,
    ql_90_test,
    model_names,
    label_txt="Evaluation Metrics Table",
    caption_txt="Evaluation Metrics Table.",
    scaling_factor=1.0,
):
    header_row = (
        "\\begin{center}\n"
        + "\captionof{table}{"
        + f"{caption_txt}"
        + "}\n"
        + "\label{"
        + f"{label_txt}"
        + "}\n"
        + "\scalebox{"
        + f"{scaling_factor}"
        + "}{\n"
        + "\\begin{tabular}{l|cccc|cccc|cccc}\n\\toprule\n\\toprule\n"
        + "&  \multicolumn{4}{c}{$\mathcal{D}_{\\text{Train}}$}"
        + "&  \multicolumn{4}{c}{$\mathcal{D}_{\\text{Validation}}$}"
        + "& \multicolumn{4}{c}{ $\mathcal{D}_{\\text{Test}}$}\\\\ \n"
        + " \cmidrule{2-5}  \cmidrule{6-9} \cmidrule{10-13}$\\text{Model}$ $\\backslash$ $\\text{Metrics}$"
        + " & NLL & CRPS & RMSE & 90\% QL & NLL & CRPS & RMSE & 90\% QL & NLL & CRPS & RMSE & 90\% QL \\\\ \\midrule"
    )
    rows = [header_row]

    for name in model_names:
        row = (
            f"{name} &  {(nlls_train[name].mean()):.4f}"
            f" &  {(crps_train[name].mean()):.4f} "
            f" & {(rmse_train[name].mean()):.4f} "
            f" & {(ql_90_train[name].mean()):.4f} "
            f" & {(nlls_val[name].mean()):.4f} "
            f" & {(crps_val[name].mean()):.4f} "
            f" & {(rmse_val[name].mean()):.4f} "
            f" & {(ql_90_val[name].mean()):.4f} "
            f" & {(nlls_test[name].mean()):.4f} "
            f" & {(crps_test[name].mean()):.4f} "
            f" & {(rmse_test[name].mean()):.4f} "
            f" & {(ql_90_test[name].mean()):.4f} \\\\ "
        )
        rows.append(row)

    table = (
        "\n".join(rows)
        + "\n\\bottomrule\n\\bottomrule"
        + "\n\\end{tabular}"
        + "\n}"
        + "\n\end{center}"
    )
    return table

def generate_latex_table_from_tidy(
    df_val: pd.DataFrame,
    df_test: pd.DataFrame,
    model_names: list[str],
    label_txt: str = "Evaluation Metrics",
    caption_txt: str = "Model comparisons based on various evaluation metrics.",
    scaling_factor: float = 1.0,
) -> str:
    """
    Build the same two‐block (Validation | Test) LaTeX table
    from tidy DataFrames produced by calculate_metrics().
    """
    def df_to_array_dict(df, metric: str):
        sub = df[df.metric == metric]
        # group by model, collect all 'value's into numpy arrays
        grouped = (
            sub
            .groupby("model")["value"]
            .apply(lambda s: s.values.astype(float))
            .to_dict()
        )
        return grouped

    nlls_val   = df_to_array_dict(df_val,   "NLL")
    crps_val   = df_to_array_dict(df_val,   "CRPS")
    rmse_val   = df_to_array_dict(df_val,   "RMSE")
    ql_90_val  = df_to_array_dict(df_val,   "QL90")

    nlls_test  = df_to_array_dict(df_test,  "NLL")
    crps_test  = df_to_array_dict(df_test,  "CRPS")
    rmse_test  = df_to_array_dict(df_test,  "RMSE")
    ql_90_test = df_to_array_dict(df_test,  "QL90")

    return generate_latex_table(
        nlls_val,
        crps_val,
        rmse_val,
        ql_90_val,
        nlls_test,
        crps_test,
        rmse_test,
        ql_90_test,
        model_names,
        label_txt=label_txt,
        caption_txt=caption_txt,
        scaling_factor=scaling_factor,
    )


def wilcoxon_by_metric(
    splits: Mapping[str, pd.DataFrame],
    baseline: str,
    metrics: Sequence[str],
    alternative: str = "less",
) -> None:
    """
    For each metric in `metrics` and each split in `splits`,
    perform a Wilcoxon signed‐rank test comparing
      baseline vs. each other model.

    Parameters
    ----------
    splits : dict of (split_name -> tidy DataFrame with columns
             ['train_size','seed_index','model','metric','value'])
    baseline : the model name to use as the reference
    metrics : list of metric names to test, e.g. ["NLL","CRPS","RMSE","QL90"]
    alternative : passed to scipy.stats.wilcoxon; default "less"
    """
    for metric in metrics:
        print(f"\n===== Metric: {metric} =====")
        for split_name, df in splits.items():
            sub = df[df.metric == metric]
            pivot = sub.pivot(
                index="seed_index", columns="model", values="value"
            )
            base = pivot[baseline].item().detach().numpy().squeeze()
            print(f"--- {split_name} ---")
            for model_name in pivot.columns:
                if model_name == baseline:
                    continue
                other = pivot[model_name].item().detach().numpy().squeeze()
                stat, p = wilcoxon(base, other, alternative=alternative)
                print(f"{baseline} < {model_name}? stat={stat:.3f}, p={p:.3e}")
                
                
def _flatten_to_1d_tensor(x):
    """Return a 1-D CPU float tensor from x which can be:
       - torch.Tensor
       - numpy array
       - list/iterable of numbers
       - list/iterable of tensors (possibly nested)
    """
    if isinstance(x, torch.Tensor):
        return x.detach().reshape(-1).to('cpu', dtype=torch.float32)

    if isinstance(x, np.ndarray):
        return torch.from_numpy(x.reshape(-1)).to('cpu', dtype=torch.float32)

    # Handle lists/iterables (possibly nested) of tensors/numbers
    if isinstance(x, (list, tuple)):
        flat = []
        stackable = True
        for xi in x:
            if isinstance(xi, torch.Tensor):
                flat.append(xi.detach().reshape(-1).cpu().float())
            elif isinstance(xi, np.ndarray):
                flat.append(torch.from_numpy(xi.reshape(-1)).cpu().float())
            elif isinstance(xi, (list, tuple)):
                # recurse once for nested lists
                flat.append(_flatten_to_1d_tensor(xi))
            else:  # number
                try:
                    flat.append(torch.tensor([float(xi)], dtype=torch.float32))
                except Exception:
                    stackable = False
        if flat and stackable:
            return torch.cat(flat).reshape(-1)
        # Fallback: try to convert whole thing at once
        return torch.tensor(list(np.array(x, dtype=float).reshape(-1)), dtype=torch.float32)

    # Fallback single number
    return torch.tensor([float(x)], dtype=torch.float32)

def row_value_mean(x):
    """Mean of the per-sample vector in x, returned as Python float."""
    t = _flatten_to_1d_tensor(x)
    return float(t.mean().item())

def row_value_scalar_for_reporting(metric, x):
    """For RMSE rows we report sqrt(mean(SE)); for others we report mean."""
    m = row_value_mean(x)
    if str(metric).upper().startswith("RMSE"):
        return float(np.sqrt(max(m, 0.0)))   # m is mean(SE), so sqrt gives RMSE
    return m


def tbl_get_mean_from_vec(df):
    df_means = df.copy()
    df_means["value_mean"] = df_means["value"].apply(row_value_mean)

    # 2) Add a "reported_mean" column where RMSE is sqrt(mean(SE))
    df_means["reported_mean"] = df_means.apply(
        lambda r: row_value_scalar_for_reporting(r["metric"], r["value"]), axis=1
    )

    # 3) If you want the same tidy schema but with scalar means in `value`:
    df_means_tidy = df.copy()
    df_means_tidy["value"] = df_means_tidy.apply(
        lambda r: row_value_scalar_for_reporting(r["metric"], r["value"]), axis=1
    )
    return(df_means_tidy)