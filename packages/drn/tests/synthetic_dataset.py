import numpy as np
import torch


def generate_synthetic_data(n=1000, seed=1):
    rng = np.random.default_rng(seed)
    x_all = rng.random(size=(n, 4))
    epsilon = rng.normal(0, 0.2, n)

    means = np.exp(
        0
        - 0.5 * x_all[:, 0]
        + 0.5 * x_all[:, 1]
        + np.sin(np.pi * x_all[:, 0])
        - np.sin(np.pi * np.log(x_all[:, 2] + 1))
        + np.cos(x_all[:, 1] * x_all[:, 2])
    ) + np.cos(x_all[:, 1])
    dispersion = 0.5

    y_lognormal = np.exp(rng.normal(means / 4, dispersion))
    y_gamma = rng.gamma(1 / dispersion, scale=dispersion * means / 4)

    y_all = y_gamma * 0.5 + y_lognormal * 0.5 + epsilon**2

    num_val = int(n * 0.2)
    num_test = int(n * 0.2)
    num_train = n - num_val - num_test

    x_train, y_train = x_all[:num_train], y_all[:num_train]
    x_val, y_val = (
        x_all[num_train : num_train + num_val],
        y_all[num_train : num_train + num_val],
    )

    return x_train, y_train, x_val, y_val


def generate_synthetic_tensordataset(n=1000, seed=1):

    x_train, y_train, x_val, y_val = generate_synthetic_data(n, seed)

    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")
    print(f"Using device {device}")

    # Convert NumPy arrays to PyTorch tensors and move them to the specified device
    X_train = torch.Tensor(x_train).to(device)
    Y_train = torch.Tensor(y_train).to(device)
    X_val = torch.Tensor(x_val).to(device)
    Y_val = torch.Tensor(y_val).to(device)

    # Create Tensor datasets for training and validation
    train_dataset = torch.utils.data.TensorDataset(X_train, Y_train)
    val_dataset = torch.utils.data.TensorDataset(X_val, Y_val)

    return X_train, Y_train, train_dataset, val_dataset
