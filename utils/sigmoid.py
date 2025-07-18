import numpy as np

# NOTE: Not in use at the moment, but might be useful later
def sigmoid(x: float) -> float:
    """
    Computes the sigmoid function for a given input x.
    :param x: Input value (can be a scalar or numpy array).
    :return: Sigmoid of x, which is 1 / (1 + exp(-x)).
    """
    return 1.0 / (1.0 + np.exp(-x))