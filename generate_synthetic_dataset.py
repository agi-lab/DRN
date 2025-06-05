import numpy as np
import pandas as pd


def generate_synthetic_gamma_lognormal(n=1000, seed=1, specific_instance=None):
    rng = np.random.default_rng(seed)
    # Parameters
    mu = [0, 0]  # means
    sigma = [0.25, 0.25]  # standard deviations
    rho = 0.25  # correlation coefficient

    # Covariance matrix
    covariance = [
        [sigma[0] ** 2, rho * sigma[0] * sigma[1]],
        [rho * sigma[0] * sigma[1], sigma[1] ** 2],
    ]

    # Generate bivariate normal distribution
    x = rng.multivariate_normal(mu, covariance, n)

    # Create a non-linear and non-stationary relationship between X_1, X_2 and Y
    means = np.exp(-x[:, 0] + x[:, 1])
    dispersion = np.exp(x[:, 0]) / (1 + np.exp((x[:, 0]) * (x[:, 1])))

    if specific_instance is not None:
        x_1 = specific_instance[0]
        x_2 = specific_instance[1]
        means = np.exp(-x_1 + x_2).repeat(n)
        dispersion = (np.exp(x_1) / (1 + np.exp(x_1 * x_2))).repeat(n)

    # Calculate the gamma and lognormal parts of the Y
    y_gamma = rng.gamma(1 / dispersion, scale=dispersion * means)
    y_lognormal = np.exp(rng.normal(np.log(means), scale=dispersion))
    # Combine the components
    y = y_gamma + y_lognormal

    return (
        pd.DataFrame(x, columns=["X_1", "X_2"]),
        pd.Series(y, name="Y"),
        means,
        dispersion,
    )


def generate_synthetic_gaussian(n=1000, seed=1, specific_instance=None):
    rng = np.random.default_rng(seed)
    # Parameters
    mu = [0, 0]  # means
    sigma = [0.5, 0.5]  # standard deviations
    rho = 0.0  # correlation coefficient

    # Covariance matrix
    covariance = [
        [sigma[0] ** 2, rho * sigma[0] * sigma[1]],
        [rho * sigma[0] * sigma[1], sigma[1] ** 2],
    ]

    # Generate bivariate normal distribution
    x = rng.multivariate_normal(mu, covariance, n)

    # Create a non-linear and non-stationary relationship between X_1, X_2 and Y
    means = -x[:, 0] + x[:, 1]  # + 0.2 * x[:, 1]**2
    dispersion = 0.5 * (x[:, 0] ** 2 + x[:, 1] ** 2)

    if specific_instance is not None:
        x_1 = specific_instance[0]
        x_2 = specific_instance[1]
        means = (-x_1 + x_2).repeat(n)
        dispersion = (0.5 * (x_1**2 + x_2**2)).repeat(n)

    y_normal = rng.normal(means, dispersion)

    # Combine the components
    y = y_normal

    return (
        pd.DataFrame(x, columns=["X_1", "X_2"]),
        pd.Series(y, name="Y"),
        means,
        dispersion,
    )


def generate_synthetic_gamma(n=1000, seed=1, specific_instance=None):
    rng = np.random.default_rng(seed)
    # Parameters
    mu = [0, 0]  # means
    sigma = [1.0, 1.0]  # standard deviations
    rho = 0.25  # correlation coefficient

    # Covariance matrix
    covariance = [
        [sigma[0] ** 2, rho * sigma[0] * sigma[1]],
        [rho * sigma[0] * sigma[1], sigma[1] ** 2],
    ]

    # Generate bivariate normal distribution
    x = rng.multivariate_normal(mu, covariance, n)

    # Create a non-linear and non-stationary relationship between X_1, X_2 and Y
    means = (
        np.exp(-x[:, 0] / 2 + x[:, 1] / 2)
        + np.abs(np.sin((x[:, 0] + x[:, 1]) * np.pi)) * 0.5
    )
    dispersion = np.exp(x[:, 0] / 3) / (1 + np.exp((x[:, 0]) * (x[:, 1])))

    if specific_instance is not None:
        x_1 = specific_instance[0]
        x_2 = specific_instance[1]
        means = (
            np.exp(-x_1 / 2 + x_2 / 2) + np.abs(np.sin((x_1 + x_2) * np.pi)) * 0.5
        ).repeat(n)
        means += means * 0.25
        dispersion = (np.exp(x_1 / 3) / (1 + np.exp(x_1 * x_2))).repeat(n)

    # Calculate the gamma and lognormal parts of the Y
    y_gamma = rng.gamma(1 / dispersion, scale=dispersion * means)

    # Combine the components
    y = y_gamma + means * 0.25

    return (
        pd.DataFrame(x, columns=["X_1", "X_2"]),
        pd.Series(y, name="Y"),
        means,
        dispersion,
    )


def generate_synthetic_complex(n=1000, p=20, seed=1, specific_instance=None):
    rng = np.random.default_rng(seed)

    # Mean and correlated inputs
    mu = np.zeros(p)
    cov = 0.25 * np.ones((p, p)) + 0.75 * np.eye(p)  # moderate correlation
    X = rng.multivariate_normal(mu, cov, n)

    # Construct nonlinear and structured relationships
    linear_part = X[:, 0] - 0.5 * X[:, 1] + 0.25 * X[:, 2]
    interaction = np.sin(X[:, 3] * X[:, 4]) + np.cos(X[:, 5])
    warped = np.tanh(X[:, 6] + X[:, 7] ** 2 - X[:, 8] * X[:, 9])

    means = np.exp(0.3 * linear_part + 0.7 * interaction) + 0.5 * warped + 1.0
    dispersion = 0.5 + 0.25 * np.abs(np.sin(X[:, 1])) + (interaction + warped) ** 2 / 5

    if specific_instance is not None:
        x_fixed = np.array(specific_instance)
        means = (
            np.exp(
                0.3 * (x_fixed[0] - 0.5 * x_fixed[1])
                + 0.7 * (np.sin(x_fixed[3] * x_fixed[4]) + np.cos(x_fixed[5]))
            )
            + 0.5 * np.tanh(x_fixed[6] + x_fixed[7] ** 2 - x_fixed[8] * x_fixed[9])
            + 1.0
        ).repeat(n)
        dispersion = (0.5 + 0.25 * np.abs(np.sin(x_fixed[1]))).repeat(n)

    print(dispersion.mean(), dispersion.std())
    # Generate distorted Gamma-like Y (strictly positive)
    y_gamma = rng.gamma(1 / dispersion, scale=dispersion * means)
    y = (
        0.25 * y_gamma
        + 0.25 * (rng.normal(means, dispersion)) ** 2
        + 0.25 * np.abs(rng.normal(size=n))
        + 1e-4
    )
    feature_names = [f"X_{i+1}" for i in range(p)]
    return (
        pd.DataFrame(X, columns=feature_names),
        pd.Series(y, name="Y"),
        means,
        dispersion,
    )
