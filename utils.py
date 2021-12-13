import numpy as np
import torch
import logging
import seaborn as sns
import matplotlib.pyplot as plt
import torchvision.datasets as datasets


def sigmoid(a):
    """
    Sigmoid Function
    """
    return 1/(1 + np.exp(-a))

def add_bias(X):
    """
    Adds bias column of ones to data X
    """
    if len(X.shape) == 1:
        return np.hstack((np.ones(len(X)).reshape(-1, 1), X.reshape(-1, 1)))
    return np.hstack((np.ones(len(X)).reshape(-1, 1), X))

def remove_bias(X_bias):
    """
    Removes bias column of ones from data X
    """
    return np.delete(X_bias, 0, axis=1)

# coloring scatterplots
col = lambda c : "red" if c == 1 else "blue"

def obtain_train_test_mnist(train_X, train_y, test_X, test_y, first_digit, second_digit):
    X_first_train = train_X[train_y == first_digit].reshape(-1, 784)
    X_second_train = train_X[train_y == second_digit].reshape(-1, 784)
    X_MNIST_train = np.vstack((X_first_train, X_second_train))
    y_MNIST_train = np.vstack((train_y[train_y == first_digit].reshape(-1, 1), train_y[train_y == second_digit].reshape(-1, 1))).flatten()

    X_first_test = test_X[test_y == first_digit].reshape(-1, 784)
    X_second_test = test_X[test_y == second_digit].reshape(-1, 784)
    X_MNIST_test = np.vstack((X_first_test, X_second_test))
    y_MNIST_test = np.vstack((test_y[test_y == first_digit].reshape(-1, 1), test_y[test_y == second_digit].reshape(-1, 1))).flatten()

    return X_MNIST_train, y_MNIST_train, X_MNIST_test, y_MNIST_test





def load_mnist_dataset(train, num_classes):
    mnist_trainset = datasets.MNIST(root='./data', train=train, download=False, transform=None)

    features, targets = mnist_trainset.data, mnist_trainset.targets

    orig_img = features.clone()

    features = features.float().div_(255.)
    features = features.reshape(features.size(0), -1)

    mask = targets.lt(num_classes)
    features = features[mask, :]
    targets = targets[mask]
    orig_img = orig_img[mask, :, :]

    features.div_(features.norm(dim=1).max())
    targets = targets.float()

    return {"features": features, "targets": targets}, orig_img

def pca(data, num_dims=None, mapping=None):
    """
    Applies PCA on the specified `data` to reduce its dimensionality to
    `num_dims` dimensions, and returns the reduced data and `mapping`.

    If a `mapping` is specified as input, `num_dims` is ignored and that mapping
    is applied on the input `data`.
    """

    # work on both data tensor and data dict:
    data_dict = False
    if isinstance(data, dict):
        assert "features" in data, "data dict does not have features field"
        data_dict = True
        original_data = data
        data = original_data["features"]
    assert data.dim() == 2, "data tensor must be two-dimensional matrix"

    # compute PCA mapping:
    if mapping is None:
        assert num_dims is not None, "must specify num_dims or mapping"
        mean = torch.mean(data, 0, keepdim=True)
        zero_mean_data = data.sub(mean)
        covariance = torch.matmul(zero_mean_data.t(), zero_mean_data)
        _, projection = torch.symeig(covariance, eigenvectors=True)
        projection = projection[:, -min(num_dims, projection.size(1)):]
        mapping = {"mean": mean, "projection": projection}
    else:
        assert isinstance(mapping, dict), "mapping must be a dict"
        assert "mean" in mapping and "projection" in mapping, "mapping missing keys"
        if num_dims is not None:
            logging.warning("Value of num_dims is ignored when mapping is specified.")

    # apply PCA mapping:
    reduced_data = data.sub(mapping["mean"]).matmul(mapping["projection"])

    # return results:
    if data_dict:
        original_data["features"] = reduced_data
        reduced_data = original_data
    return reduced_data, mapping


def line_plot(
        Y, X, xlabel=None, ylabel=None, ymax=None, ymin=None,
        xmax=None, xmin=None, filename=None, legend=None, errors=None,
        xlog=False, ylog=False, size=None, marker="s"):
    colors = sns.cubehelix_palette(Y.shape[0], start=2, rot=0, dark=0, light=.5)
    plt.clf()
    if legend is None:
        legend = [None] * Y.shape[0]

    if size is not None:
        plt.figure(figsize=size)

    for n in range(Y.shape[0]):
        x = X[n, :] if X.ndim == 2 else X
        plt.plot(x, Y[n, :], label=legend[n], color=colors[n],
                marker=marker, markersize=5)
        if errors is not None:
            plt.fill_between(
                x, Y[n, :] - errors[n, :], Y[n, :] + errors[n, :],
                alpha=0.1, color=colors[n])

    if ymax is not None:
        plt.ylim(top=ymax)
    if ymin is not None:
        plt.ylim(bottom=ymin)
    if xmax is not None:
        plt.xlim(right=xmax)
    if xmin is not None:
        plt.xlim(left=xmin)

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if legend[0] is not None:
        plt.legend()

    axes = plt.gca()
    if xlog:
        axes.semilogx(10.)
    if ylog:
        axes.semilogy(10.)

    if filename is not None:
        plt.savefig(filename, dpi=1200, bbox_inches="tight")
