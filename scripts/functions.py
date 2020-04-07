import numpy as np


def sigmoid(x, derivate=False):
    """
    Compute the element wise sigmoid for the array x
    Args:
        x ([type]): [description]
        derivate (bool, optional): [description]. Defaults to False.
    """
    x = x + 1e-12  # why are we doing this?
    f = 1/(1 + np.exp(x))
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

    f = (np.exp(x) - np.exp(-x))/(np.exp(x) + np.exp(-x))
    if derivate:
        return 1 - f**2
    else:
        return f


def softmax(f, derivate=False):
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
