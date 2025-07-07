import numpy as np

def sigmoid(x: float) -> float:
    """
    Computes the sigmoid function for a given input x.
    
    Parameters:
        x (float): The input value to the sigmoid function.
        
    Returns:
        float: The output of the sigmoid function, which is in the range (0, 1).
    """
    return 1.0 / (1.0 + np.exp(-x))