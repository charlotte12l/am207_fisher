import numpy as np
import torch

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