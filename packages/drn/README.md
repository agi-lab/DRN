# drn - A Python Package for Distributional Refinement Network (DRN)

## Minimal working example

This section provides a minimal working example of how to use the `drn` package to train a Distributional Refinement Network (DRN) on a synthetic dataset.

```python
from drn import GLM, DRN
import pandas as pd

DATA_URL = "https://raw.githubusercontent.com/agi-lab/DRN/refs/heads/main/data/processed/synth/"
X_train = pd.read_csv(DATA_URL + "x_train.csv")
y_train = pd.read_csv(DATA_URL + "y_train.csv")

glm_model = GLM("gamma").fit(X_train, y_train)
drn_model = DRN(glm_model).fit(X_train, y_train)
```

Read on for more details on how to use the `drn` package, including installation instructions, examples of training a DRN, and how to interpret the results.

## Table of Contents  
- [Overview](#overview) 
- [Key Features](#key-features)
- [Installation](#installation)
- [Example: Train a DRN](#example-train-a-drn)
    - [DRN Baseline Component](#drn-baseline-component)
    - [DRN Deep Learning Component](#drn-deep-learning-component)
- [Example: Distributional Forecasts and Interpretability](#example-distributional-forecasts-and-interpretability)
    - [Distributional Properties: Mean and Quantiles](#distributional-properties-mean-and-quantiles)
    - [Forecasting Performance: Evaluation Metrics](#forecasting-performance-evaluation-metrics)
    - [Interpretability: Kernel SHAP-Embedded PDF and CDF](#interpretability-kernel-shap-embedded-pdf-and-cdf)
- [Related Repository](#related-repository)
- [License](#license) 
- [Authors](#authors)
- [Citations](#citation)
- [Contact](#Contact)

## Overview

A key task in actuarial modelling involves modelling the distributional properties of losses. Classic (distributional) regression approaches like Generalized Linear Models (GLMs; Nelder and Wedderburn, 1972) are commonly used, but challenges remain in developing models that can:
1. Allow covariates to flexibly impact different aspects of the conditional distribution,
2. Integrate developments in machine learning and AI to maximise the predictive power while considering (1), and,
3. Maintain a level of interpretability in the model to enhance trust in the model and its outputs, which is often compromised in efforts pursuing (1) and (2).

We tackle this problem by proposing a Distributional Refinement Network (DRN), which combines an inherently interpretable baseline model (such as GLMs) with a flexible neural network--a modified Deep Distribution Regression (DDR; Li et al., 2021) method.
Inspired by the Combined Actuarial Neural Network (CANN; Schelldorfer and W{\''u}thrich, 2019), our approach flexibly refines the entire baseline distribution. 
As a result, the DRN captures varying effects of features across all quantiles, improving predictive performance while maintaining adequate interpretability.
 
This package, `drn`, addresses the challenges listed above and yields the results demonstrated in our [DRN paper](https://arxiv.org/abs/2406.00998) (Avanzi et al. 2024).
The full range of key features, installation procedure, examples, and related repositories are listed in the following sections.

## Key Features 

- **Comprehensive Distributional Regression Models**: 
The `drn` package includes advanced distributional regression models such as the Distributional Refinement Network (DRN), Combined Actuarial Neural Network, Mixture Density Network (MDN; Bishop, 1994), and Deep Distribution Regression (DDR). 
Built on PyTorch, it offers a user-friendly neural network training framework with features like early stopping, dropout, and other essential functionalities.

- **Exceptional Distributional Flexibility with Tailored Regularisation**: 
The DRN can accurately model the entire distribution for forecasting purposes. 
Users can control the range and extent of the baseline refinement across all quantiles. 
The recommended baseline model for DRN is a GLM. 
However, in theory, the baseline can be any form of distributional regression method, accommodating bounded, unbounded, discrete, continuous, or mixed response variables. 
The balance between the baseline and the deep learning component is regulated by the KL divergence between them, with user-defined directionality.
The smoothness of the final forecast density can be adjusted using a roughness penalty, both of which can be tuned for more precise and reliable distributional flexibility.

- **Full Distributional Forecasting and Various Evaluation Metrics**:
The regression models provide full distributional forecasting information, including density, cumulative density function, mean, and quantiles. 
The package includes a range of metrics for evaluating forecasting performance, such as Root Mean Squared Error (RMSE), Quantile Loss (QL), Continuous Ranked Probability Score (CRPS), and Negative Log-Likelihood (NLL). 
These metrics enable a comprehensive assessment of the model's performance across different aspects of distributional forecasting.

- **Reasonable Distributional Interpretability with Integrated Kernel SHAP Analysis**: 
The recommended baseline model for DRN is a GLM due to its inherent interpretability, as discussed in the [DRN paper](https://arxiv.org/abs/2406.00998). 
Additional, DRN integrates interpretability techniques like SHAP, allowing users to see detailed decomposition of contributions from both the baseline model and the DRN across various distributional properties beyond the mean. 
Users can generate plots for both density and CDF for the baseline and refined models. 
Kernel SHAP analysis is embedded within these plots, providing customised post-hoc interpretability and aiding in understanding the model's adjustments of key distributional properties beyond the mean.


## Installation

To install the DRN package, simply run:

```sh
pip install drn
```

## Example: Train a DRN

This section demonstrates how to construct DRN using our `drn` package from scratch.
After loading all relevant packages, we generate a synthetic Gaussian dataset.

``` python
from drn import train, split_and_preprocess
from drn import GLM, DRN
from drn import models
import numpy as np
import pandas as pd
import torch
```

``` python
def generate_synthetic_gaussian_lognormal(n=1000, seed=1, specific_instance=None):
    rng = np.random.default_rng(seed)
    
    # Parameters
    mu = [0, 0]  # Means of the Gaussian
    sigma = [0.5, 0.5]  # Standard deviations
    rho = 0.0  # Correlation coefficient

    # Covariance matrix
    covariance = [
        [sigma[0] ** 2, rho * sigma[0] * sigma[1]],
        [rho * sigma[0] * sigma[1], sigma[1] ** 2],
    ]

    # Generate bivariate normal distribution
    x = rng.multivariate_normal(mu, covariance, n)

    # Create a non-linear relationship between X1 & X2 and means & dispersion.  
    means = -x[:, 0] + x[:, 1]
    dispersion = 0.5 * (x[:, 0] ** 2 + x[:, 1] ** 2)

    # Use specific instance if provided
    if specific_instance is not None:
        x_1, x_2 = specific_instance
        means = (-x_1 + x_2).repeat(n)
        dispersion = (0.5 * (x_1 ** 2 + x_2 ** 2)).repeat(n)

    # Generate response variable Y, which consists both normal and lognormal components
    y_normal = rng.normal(means, dispersion)
    y_lognormal = np.exp(rng.normal(np.log(means**2), scale = dispersion))
    y = y_normal - y_lognormal

    return pd.DataFrame(x, columns=["X_1", "X_2"]), pd.Series(y, name="Y")

# Generate synthetic data
features, target = generate_synthetic_gaussian_lognormal(12000)
```

You can choose to split and preprocess the dataset as you wish. 
The following is just an example to generate a training and validation dataset compatible for training using PyTorch.

``` python
import torch

# Preprocess and split the data
x_train, x_val, x_test, y_train, y_val, y_test, \
x_train_raw, x_val_raw, x_test_raw, \
num_features, cat_features, all_categories, ct = split_and_preprocess(
    features,
    target,
    ['X_1', 'X_2'],  # Numerical features
    [],  # Categorical features
    seed=0,
    num_standard=True  # Whether to standardize or not
)

# Convert pandas dataframes to PyTorch tensors
X_train = torch.Tensor(x_train.values)
Y_train = torch.Tensor(y_train.values)
X_val = torch.Tensor(x_val.values)
Y_val = torch.Tensor(y_val.values)
X_test = torch.Tensor(x_test.values)
Y_test = torch.Tensor(y_test.values)

# Create PyTorch datasets for training and validation
train_dataset = torch.utils.data.TensorDataset(X_train, Y_train)
val_dataset = torch.utils.data.TensorDataset(X_val, Y_val)
```

### DRN Baseline Component

The first stage of constructing a DRN involves training a baseline distributional regression model, such as a GLM.
Below, we use the GLM from statsmodels.
You don't need to add the intercept term, just pass in X_train and Y_train as torch tensors.

``` python
baseline = GLM('gaussian')
baseline.fit(X_train, Y_train)
``` 

Alternatively, you can train a GLM using the SGD method.
We currently support the 'gaussian' and 'gamma' distributions for the neural network version of GLM.

``` python
# Initialise and train the baseline GLM model
torch.manual_seed(23)
baseline = GLM('gaussian')
baseline.fit(X_train, Y_train, grad_descent=True, 
             epochs=5000, lr=0.001, patience=100, batch_size=100)
```

### DRN Deep Learning Component

You need to first specify a region for distributional refinement of the baseline, defined by `cutpoints_DRN`.
Select a lower bound `c_0`, an upper bound `c_K`, a proportion `p` (cutpoints-to-observation ratio) and the minumum number of training observations `min_obs` needed for each partitioned interval.
In practice:
- Try `p` around 0.05-0.1 for small datasets (less than 10000 observations) and decrease `p` as the number of observations increases.
- Try `min_obs` = 0 for small datasets and increase `min_obs` as the number of training observations increases, if desirable.

``` python
# Define the refinement region for DRN
cutpoints_DRN = models.drn_cutpoints(
    c_0 = np.min(y_train) * 1.1 if np.min(y_train) < 0 else 0.0,
    c_K = np.max(y_train) * 1.1,
    p = 0.1,
    y = y_train,
    min_obs = 1
)
```

Finally, specify the hyperparameters for the DRN and pass in the GLM and refinement region defined earlier.
The regularisation coefficients `kl_alpha`, `dv_alpha`, and `mean_alpha` control the deviation from the baseline's distribution, the roughness of the estimated density, and the deviation from the baseline's mean, respectively.
All of these coefficients can be treated as hyperparameters. 
Nevertheless:
- Try a small `kl_alpha`, i.e., 1e-5~1e-4, depending on the performance of the baseline (generally, the better the baseline, the larger the `kl_alpha`).
- Try a reasonably large `dv_alpha` for a small number of cutpoints, i.e., ~1e-3. Decrease `dv_alpha` as the number of cutpoints increases.
- Try to start with a small `mean_alpha`, i.e., 1e-5~1e-4. Alternatively, set it to zero if total deviations from the baseline's means are ideal.

``` python
# Initialise and train the DRN model
torch.manual_seed(23)
drn_model = DRN(
    baseline=baseline,
    cutpoints=cutpoints_DRN,
    hidden_size=128,
    num_hidden_layers=2,
    baseline_start=False,
    dropout_rate=0.2
)

train(
    drn_model,
    lambda pred, y: models.drn_loss(
        pred,
        y,
        kl_alpha=1e-4,  # KL divergence penalty
        dv_alpha=1e-3,  # Roughness penalty
        mean_alpha=1e-5,  # Mean penalty
        kl_direction = 'forwards'
    ),
    train_dataset,
    val_dataset,
    lr=0.0005,
    batch_size=256,
    log_interval=1,
    patience=30,
    epochs=1000
)

drn_model.eval()
```


## Example: Distributional Forecasts and Interpretability

This section demonstrates how to use a DRN to forecast probability density functions (PDFs) and cumulative density functions (CDFs), key distributional properties, and evaluate distributional forecasting performance.

``` python
from drn import metrics
from drn import interpretability
```

### Distributional Properties: Mean and Quantiles

Currently, we support mean and quantile forecasts. 
Variance, skewness, and kurtosis can be derived using the density function.

``` python
test_instance = X_test[:1]
mean_pred = drn_model.predict(test_instance).mean
_10_quantile = drn_model.predict(test_instance).quantiles(
      [10],
      l = torch.min(Y_train) * 3 if torch.min(Y_train) < 0 else 0.0,
      u = torch.max(Y_train) * 3)
_90_quantile = drn_model.predict(test_instance).quantiles(
      [90],
      l = torch.min(Y_train) * 3 if torch.min(Y_train) < 0 else 0.0,
      u = torch.max(Y_train) * 3)
_99_quantile = drn_model.predict(test_instance).quantiles(
      [99],
      l = torch.min(Y_train) * 3 if torch.min(Y_train) < 0 else 0.0,
      u = torch.max(Y_train) * 3)

for metric_name, metric in zip(['Mean', '10% Quantile', '90% Quantile', '99% Quantile'],
                               [mean_pred, _10_quantile, _90_quantile, _99_quantile]):
    print(f'{metric_name}: {metric.item()}')
```

### Forecasting Performance: Evaluation Metrics

Generate both the `distributions` and the `cdf` objects.

``` python
names = ["GLM", "DRN"]
dr_models = [baseline, drn_model]

print("Generating distributional forecasts")
dists_train, dists_val, dists_test = {}, {}, {}
for name, model in zip(names, dr_models):
    print(f"- {name}")
    dists_train[name] = model.predict(X_train)
    dists_val[name] = model.predict(X_val)
    dists_test[name] = model.predict(X_test)

print("Calculating CDF over a grid")
GRID_SIZE = 3000
grid = torch.linspace(0, np.max(y_train) * 1.1, GRID_SIZE).unsqueeze(-1)

cdfs_train, cdfs_val, cdfs_test = {}, {}, {}
for name, model in zip(names, dr_models):
    print(f"- {name}")
    cdfs_train[name] = dists_train[name].cdf(grid)
    cdfs_val[name] = dists_val[name].cdf(grid)
    cdfs_test[name] = dists_test[name].cdf(grid)
```

Then, generate the evaluation metrics: NLL, CRPS, RMSE, and QLs.

``` python
print("Calculating negative log likelihoods")
nlls_train, nlls_val, nlls_test = {}, {}, {}
for name, model in zip(names, dr_models):
    nlls_train[name] = -dists_train[name].log_prob(Y_train).mean()
    nlls_val[name] = -dists_val[name].log_prob(Y_val).mean()
    nlls_test[name] = -dists_test[name].log_prob(Y_test).mean()

for nll_dict, df_name in zip([nlls_train, nlls_val, nlls_test], ['training', 'val', 'test']):
    print(f'NLL on {df_name} set')
    for name in names:
        print(f"{name}: {nll_dict[name]:.4f}")
    print('-------------------------------')

print("Calculating CRPS")
grid = grid.squeeze()
crps_train, crps_val, crps_test = {}, {}, {}
for name, model in zip(names, dr_models):
    crps_train[name] = metrics.crps(Y_train, grid, cdfs_train[name])
    crps_val[name] = metrics.crps(Y_val, grid, cdfs_val[name])
    crps_test[name] = metrics.crps(Y_test, grid, cdfs_test[name])

for crps_dict, df_name in zip([crps_train, crps_val, crps_test], ['training', 'val', 'test']):
    print(f'CRPS on {df_name} set')
    for name in names:
        print(f"{name}: {crps_dict[name].mean():.4f}")
    print('------------------------------')

print("Calculating RMSE")
rmse_train, rmse_val, rmse_test = {}, {}, {}
for name, model in zip(names, dr_models):
    means_train = dists_train[name].mean
    means_val = dists_val[name].mean
    means_test = dists_test[name].mean
    rmse_train[name] = metrics.rmse(y_train, means_train)
    rmse_val[name] = metrics.rmse(y_val, means_val)
    rmse_test[name] = metrics.rmse(y_test, means_test)

for rmse_dict, df_name in zip([rmse_train, rmse_val, rmse_test], ['training', 'validation', 'test']):
    print(f'RMSE on {df_name} set')
    for name in names:
        print(f"{name}: {rmse_dict[name].mean():.4f}")
    print('-------------------------------')

print("Calculating Quantile Loss")
ql_90_train, ql_90_val, ql_90_test = {}, {}, {}
for features, response, dataset_name, ql_dict in zip(
    [X_train, X_val, X_test], [y_train, y_val, y_test], ['Training', 'Validation', 'Test'], [ql_90_train, ql_90_val, ql_90_test]
):
    print(f'{dataset_name} Dataset Quantile Loss(es)')
    for model, model_name in zip(dr_models, names):
        ql_dict[model_name] = metrics.quantile_losses(
            0.9, model, model_name, features, response,
            max_iter=1000, tolerance=1e-4,
            l=torch.Tensor([np.min(y_train) - 3 * (np.max(y_train) - np.min(y_train))]),
            u=torch.Tensor([np.max(y_train) + 3 * (np.max(y_train) - np.min(y_train))])
        )
    print('----------------------')

```

### Interpretability: Kernel SHAP-Embedded PDF and CDF

To plot the PDF and CDF, you need to first initialise an `explainer`. 
The instance to be examined should be a DataFrame, with feature values that are neither standardised nor encoded.
We currently support the Kernel SHAP method.

``` python
test_instance_df = x_test_raw.iloc[:1]
Y_instance = Y_test[:1]

# Initialise the Explainer
drn_explainer = interpretability.DRNExplainer(
                            drn_model,
                            baseline,
                            cutpoints_DRN,
                            x_train_raw,
                            cat_features,
                            all_categories,
                            ct
                            )  

# Plot the PDF before and after refinement
drn_explainer.plot_adjustment_factors(
                                    instance=test_instance_df,
                                    num_interpolations=1_000,
                                    plot_adjustments_labels=False,
                                    x_range=(-2, 2),
                                    )

# Use Kernel SHAP to explain the mean adjustment
drn_explainer.plot_dp_adjustment_shap(
                                    instance_raw=test_instance_df,
                                    method='Kernel',
                                    nsamples_background_fraction=0.5,
                                    top_K_features=2,
                                    labelling_gap=0.1,
                                    dist_property='Mean',
                                    x_range=(-1, 1),
                                    y_range=(0.0, 2.0),
                                    observation=Y_instance,
                                    adjustment=True,
                                    shap_fontsize=15,
                                    figsize=(7, 7),
                                    plot_title='Explaining a 90% Quantile Adjustment',
                                    )

# Explain DRN's 90% quantile prediction from ground up
drn_explainer.cdf_plot(
                    instance=test_instance_df,
                    method='Kernel',
                    nsamples_background_fraction=0.5,
                    top_K_features=2,
                    labelling_gap=0.1,
                    dist_property='90% Quantile',
                    x_range=(-0.5, 1.0),
                    y_range=(0.8, 1.0),
                    adjustment=False,
                    plot_baseline=False,
                    shap_fontsize=15,
                    figsize=(7, 7),
                    plot_title='90% Quantile Explanation',
                    )
```

## Related Repository

This package accompanies the [DRN paper](https://arxiv.org/abs/2406.00998) on the Distributional Refinement Network (DRN).
The related repository, available at [https://github.com/agi-lab/DRN](https://github.com/agi-lab/DRN), contains the Python notebooks and additional resources needed to reproduce the results presented in the [DRN paper](https://arxiv.org/abs/2406.00998).

## License 

See [LICENSE.md](https://github.com/EricTianDong/drn/tree/main?tab=MIT-1-ov-file).

## Authors

- Eric Dong (author, maintainer),
- Patrick Laub (author).


## Citation

``` sh
@misc{avanzi2024distributional,
      title={Distributional Refinement Network: Distributional Forecasting via Deep Learning}, 
      author={Benjamin Avanzi and Eric Dong and Patrick J. Laub and Bernard Wong},
      year={2024},
      eprint={2406.00998},
      archivePrefix={arXiv},
      primaryClass={stat.ML}
}
```

## Contact

For any questions or further information, please contact tiandong1999@gmail.com.


