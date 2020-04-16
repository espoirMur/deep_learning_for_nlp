import numpy as np


def sigmoid(x, derivate=False):
    """
    Compute the element wise sigmoid for the array x
    Args:
        x ([type]): [description]
        derivate (bool, optional): [description]. Defaults to False.
    """
    x = x + 1e-12  # why are we doing this?
    f = 1 / (1 + np.exp(x))
    if derivate:
        f = f * (1 - f)
    else:
        return f


def tanh(x, derivate=False):
    """
    Compute tanh function

    Args:
        x ([type]): [description]
        derivate (bool, optional): [description]. Defaults to False.
    """
    x = x + 1e-12

    f = (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))
    if derivate:
        return 1 - f**2
    else:
        return f


def softmax(x, derivate=False):
    """
    Compute softmax derivate of x

    Args:
        f ([type]): [description]
        derivate (bool, optional): [description]. Defaults to False.
    """
    x = x + 1e-12

    f = np.exp(x) / np.sum(np.exp(x))

    if derivate:
        pass
    else:
        return f


def cross_entropy(predictions, targets, epsilon=1e-12):
    """
    Computes cross entropy between targets (encoded as one-hot vectors)
    and predictions.
    Input: predictions (N, k)
           targets (N, k)
    Returns: scalar
    """
    predictions = np.clip(predictions, epsilon, 1. - epsilon)
    cross_entropy = -np.mean(targets * np.log(predictions + 1e-9))
    return cross_entropy


def log_loss(y_predicted, y, derivate=False):
    """
    Compute the log loss between the predicted output and the real output
    Args:
        y_predicted ([type]): [description]
        y ([type]): [description]
        derivate (bool, optional): [description]. Defaults to False.
    """
    f = - y * np.log(y_predicted)

    if derivate:
        return - y / y_predicted
    else:
        return f
