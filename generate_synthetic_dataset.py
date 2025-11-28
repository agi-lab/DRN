import numpy as np
import pandas as pd


def generate_synthetic_gamma_lognormal(n=1000, seed=1, specific_instance=None, version = 'old'):
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

    if version == 'old':
        # Create a non-linear and non-stationary relationship between X_1, X_2 and Y
        means = np.exp(-x[:, 0] + x[:, 1]) 
        dispersion = np.exp(x[:, 0]) / (1 + np.exp((x[:, 0]) * (x[:, 1])))
    else: 
        # Create a non-linear and non-stationary relationship between X_1, X_2 and Y
        means = np.exp(-x[:, 0] + x[:, 1]) + np.abs(np.sin((x[:, 0] * 2 + x[:, 1] * 2) * np.pi)) 
        dispersion = 0.25 + np.exp(x[:, 0]) / (1 + np.exp((x[:, 0]) * (x[:, 1])))

    if specific_instance is not None:
        x_1 = specific_instance[0]
        x_2 = specific_instance[1]
        if version == 'old':
            means = (np.exp(-x_1 + x_2)).repeat(n)
            dispersion = (np.exp(x_1) / (1 + np.exp(x_1 * x_2))).repeat(n)
        else:
            means = (np.exp(-x_1 + x_2) + np.abs(np.sin((x_1 * 2 + x_2 * 2) * np.pi))).repeat(n)
            dispersion = (0.25+np.exp(x_1) / (1 + np.exp(x_1 * x_2))).repeat(n)

    # Calculate the gamma and lognormal parts of the Y
    y_gamma = rng.gamma(1 / dispersion, scale=dispersion * means)
    y_lognormal = np.exp(rng.normal(np.log(means), scale=dispersion))
    # Combine the components
    y = (y_gamma) + (y_lognormal)
    print(y_gamma.mean(), y_lognormal.mean())

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


def generate_synthetic_baseline_dependency(n=1000, seed=1, specific_instance=None, version = 'ds3'):
    rng = np.random.default_rng(seed)
    # Parameters
    mu = [0, 0]  # means
    sigma = [1.0, 1.0]  # standard deviations
    rho = 0.1  # correlation coefficient

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

    means_1 = (
            (-x[:, 0] / 2 + x[:, 1] / 2)
            + np.sin((x[:, 0] + x[:, 1]) * np.pi) 
        )
    dispersion_1 = np.exp(x[:, 0]*x[:, 1]/2) / (1 + np.exp((x[:, 0]) * (x[:, 1]))) 

    means_2 = 0.1 + x[:, 0]/2 + x[:, 1]/ 2
    dispersion_2 = np.tanh(x[:, 0] + x[:, 1]) ** 2 + 0.25 #np.exp(x[:, 0]*x[:, 1]/2) / (1 + np.exp((x[:, 0]) + (x[:, 1])))
    
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
    y_gamma_1 = rng.gamma(1 / dispersion_1, scale=dispersion_1 * (means_1)**2)
    y_gamma_2 = rng.gamma(1 / dispersion_2, scale=dispersion_2 * np.exp(means_2))
    y_abs_1 = np.abs(rng.normal(means_1, scale=dispersion_1, size=n)) 
    y_abs_2 = np.abs(rng.normal(means_2, scale=dispersion_2, size=n))
    
    # Combine the components
    # y = y_gamma + means * 0.25
    
    if version == 'ds1':
        y = (y_abs_1+y_abs_2)**(3)
    elif version == 'ds2':
        y = np.log(y_gamma*y_gamma_1*y_gamma_2)
    else:    
        y = y_gamma + y_abs_1**3
        
    return (
        pd.DataFrame(x, columns=["X_1", "X_2"]),
        pd.Series(y, name="Y"),
        means,
        dispersion,
    )

