import matplotlib.pyplot as plt
import numpy as np
import scipy
import statsmodels.api as sm
import torch
from drn import crps
from scipy.stats import wilcoxon
from tqdm.auto import trange

# from drn import *


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


def quantile_points(cdfs, response, grid):
    response = np.array(response)
    GLM_points = [[0]] * len(response)
    CANN_points = [[0]] * len(response)
    MDN_points = [[0]] * len(response)
    DDR_points = [[0]] * len(response)
    DRN_points = [[0]] * len(response)

    for k in trange(len(response)):
        # GLM
        GLM_points[k] = quantile_residuals(
            response[k], cdfs["GLM"][:, k].detach().numpy(), grid.detach().numpy()
        )

        # CANN
        CANN_points[k] = quantile_residuals(
            response[k], cdfs["CANN"][:, k].detach().numpy(), grid.detach().numpy()
        )

        # MDN
        MDN_points[k] = quantile_residuals(
            response[k], cdfs["MDN"][:, k].detach().numpy(), grid.detach().numpy()
        )

        # DDR
        DDR_points[k] = quantile_residuals(
            response[k], cdfs["DDR"][:, k].detach().numpy(), grid.detach().numpy()
        )

        # DRN
        DRN_points[k] = quantile_residuals(
            response[k], cdfs["DRN"][:, k].detach().numpy(), grid.detach().numpy()
        )

    return (GLM_points, MDN_points, DDR_points, DRN_points, CANN_points)


def quantile_residuals_plots(model_points):
    quantiles = [0] * len(model_points)
    for i in range(len(model_points)):
        quantiles[i] = np.array(scipy.stats.norm.ppf(model_points[i]))

    figure, axes = plt.subplots(2, 2, figsize=(26, 26))
    sm.qqplot(quantiles[0], line="45", ax=axes[0, 0])
    sm.qqplot(quantiles[1], line="45", ax=axes[0, 1])
    sm.qqplot(quantiles[2], line="45", ax=axes[1, 0])
    sm.qqplot(quantiles[3], line="45", ax=axes[1, 1])
    # sm.qqplot(quantiles[4], line="45", ax=axes[2, 0])
    axes[0, 0].set_title("GLM", fontsize=45, color="black")
    axes[0, 0].set_ylim(-4, 4)
    axes[0, 0].set_xlim(-3.2, 3.2)
    axes[0, 1].set_title("MDN", fontsize=45, color="black")
    axes[0, 1].set_ylim(-4, 4)
    axes[0, 1].set_xlim(-3.2, 3.2)
    axes[1, 0].set_title("DDR", fontsize=45, color="black")
    axes[1, 0].set_ylim(-4, 4)
    axes[1, 0].set_xlim(-3.2, 3.2)
    axes[1, 1].set_title("DRN", fontsize=45, color="black")
    axes[1, 1].set_ylim(-4, 4)
    axes[1, 1].set_xlim(-3.2, 3.2)

    # Set font size for all axes labels and tick labels
    for ax in axes.flat:
        # Set the font size of axis labels
        ax.set_xlabel("Theoretical Quantiles", fontsize=45)  # Adjust fontsize as needed
        ax.set_ylabel("Sample Quantiles", fontsize=45)  # Adjust fontsize as needed

        # Set the font size of tick labels
        ax.tick_params(
            axis="both", which="major", labelsize=40
        )  # Adjust labelsize as needed

    figure.suptitle("Quantile Residuals", fontsize=60, y=0.99)  # fontweight="bold"
    plt.tight_layout(pad=2)


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
        Q_predicted[k] = np.array(cdfs_[k])[closest_index(y, grid)]
        all_quantiles = [
            np.array(cdfs_[j])[closest_index(responses[j], grid)] <= Q_predicted[k]
            for j in range(len(responses))
        ]
        Q_empirical[k] = np.sum(all_quantiles) / len(responses)

    return (Q_predicted, Q_empirical)


def calibration_plot(cdfs_, y, grid):
    responses = np.array(y)

    Q_predicted_GLM, Q_empirical_GLM = calibration_plot_stats(
        cdfs_["GLM"].detach().numpy(), grid.detach().numpy(), responses
    )
    Q_predicted_CANN, Q_empirical_CANN = calibration_plot_stats(
        cdfs_["CANN"].detach().numpy(), grid.detach().numpy(), responses
    )
    Q_predicted_MDN, Q_empirical_MDN = calibration_plot_stats(
        cdfs_["MDN"].detach().numpy(), grid.detach().numpy(), responses
    )
    Q_predicted_DDR, Q_empirical_DDR = calibration_plot_stats(
        cdfs_["DDR"].detach().numpy(), grid.detach().numpy(), responses
    )
    Q_predicted_DRN, Q_empirical_DRN = calibration_plot_stats(
        cdfs_["DRN"].detach().numpy(), grid.detach().numpy(), responses
    )

    GLM_STATS = np.sum(
        (np.array(Q_predicted_GLM) - np.array(Q_empirical_GLM)) ** 2
    ) / len(responses)
    CANN_STATS = np.sum(
        (np.array(Q_predicted_CANN) - np.array(Q_empirical_CANN)) ** 2
    ) / len(responses)
    MDN_STATS = np.sum(
        (np.array(Q_predicted_MDN) - np.array(Q_empirical_MDN)) ** 2
    ) / len(responses)
    DDR_STATS = np.sum(
        (np.array(Q_predicted_DDR) - np.array(Q_empirical_DDR)) ** 2
    ) / len(responses)
    DRN_STATS = np.sum(
        (np.array(Q_predicted_DRN) - np.array(Q_empirical_DRN)) ** 2
    ) / len(responses)

    figure, axes = plt.subplots(1, 1, figsize=(10, 10))

    plt.scatter(
        Q_predicted_GLM,
        Q_empirical_GLM,
        s=6,
        color="gray",
        label=f"GLM \n $\sum_j (p_j-\hat p_j)^2= {round(GLM_STATS*len(responses), 4)}$",
    )
    plt.scatter(
        Q_predicted_CANN,
        Q_empirical_CANN,
        s=6,
        color="green",
        label=f"CANN \n $\sum_j (p_j-\hat p_j)^2= {round(CANN_STATS*len(responses), 4)}$",
    )
    plt.scatter(
        Q_predicted_MDN,
        Q_empirical_MDN,
        s=6,
        color="black",
        label=f"MDN \n $\sum_j (p_j-\hat p_j)^2= {round(MDN_STATS*len(responses), 4)}$",
    )
    plt.scatter(
        Q_predicted_DDR,
        Q_empirical_DDR,
        s=6,
        label=f"DDR \n $\sum_j (p_j-\hat p_j)^2= {round(DDR_STATS*len(responses), 4)}$",
    )
    plt.scatter(
        Q_predicted_DRN,
        Q_empirical_DRN,
        s=6,
        color="red",
        label=f"DRN  \n $\sum_j (p_j- p_j)^2={round(DRN_STATS*len(responses), 4)}$",
    )
    plt.plot([0, 1], [0, 1], ls="--", color="red")
    plt.xlabel("Predicted: $\hat{p}$", fontsize=30)
    plt.ylabel("Empirical: $p$", fontsize=30)
    plt.title("Calibration Plot", fontsize=30)
    legend = plt.legend(
        prop={"size": 15}, scatterpoints=1
    )  # Increase scatterpoints for larger marker
    for handle in legend.legend_handles:
        handle.set_sizes([40])


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


# +


def quantile_score_raw(y_true, y_pred, p):
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
    return torch.where(y_true >= y_pred, p * e, (1 - p) * -e)


def quantile_losses_raw(
    p,
    model,
    model_name,
    X,
    y,
    max_iter=1000,
    tolerance=5e-5,
    l=None,
    u=None,
    print_score=True,
):
    """
    Calculate and optionally print the quantile loss for the given data and model.

    :param p: The cumulative probability ntile as a float
    :param model: The trained model.
    :param model_name: The name of the trained model.
    :param X: Input features as a Pandas DataFrame or numpy array.
    :param y: True target values as a Pandas Series or numpy array.
    :param max_iter: The maximum number of iterations for the quantile search algorithm.
    :param tolerance: The tolerance for convergence of the the quantile search algorithm.
    :param l: The lower bound for the quantile search
    :param u: The upper bound for the quantile search
    :param print_score: A boolean indicating whether to print the score.
    :return: The quantile loss as a PyTorch tensor.
    """
    # Predict quantiles based on the model name
    if model_name in ["GLM", "MDN", "CANN"]:
        predicted_quantiles = model.quantiles(
            X, [p * 100], max_iter=max_iter, tolerance=tolerance, l=l, u=u
        )
    elif model_name in ["DDR", "DRN"]:
        predicted_quantiles = model.distributions(X).quantiles(
            [p * 100], max_iter=max_iter, tolerance=tolerance, l=l, u=u
        )

    # Compute the quantile score
    score = quantile_score_raw(y, predicted_quantiles, p)

    return score


def ql90_wilcoxon_test(models, X_features, Y_target, y_train, dataset="Test"):
    glm, cann, mdn, ddr, drn = models

    # 90% QL data
    ql_glm = (
        quantile_losses_raw(
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
        quantile_losses_raw(
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
        quantile_losses_raw(
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
        quantile_losses_raw(
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
        quantile_losses_raw(
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
