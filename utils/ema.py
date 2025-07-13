import numpy as np

def apply_asymmetric_ema(prev: float, new: float, alpha_up: float, alpha_down: float) -> float:
    """
    Apply asymmetric EMA to a single value.
    :param prev: Previous value.
    :param new: New value.
    :param alpha_up: Alpha for increases.
    :param alpha_down: Alpha for decreases.
    :return: New value after applying EMA.
    """
    if new > prev:
        return alpha_up * new + (1 - alpha_up) * prev
    else:
        return alpha_down * new + (1 - alpha_down) * prev
    
# NOTE: No longer in use at the moment, but might by useful later
def apply_background_color_asymmetric_ema(prev: np.ndarray, new: np.ndarray, alpha_up: float, alpha_down: float) -> np.ndarray: 
    """
    Apply asymmetric EMA to background color.
    :param prev: Previous background color.
    :param alpha_up: Alpha for increases.
    :param alpha_down: Alpha for decreases.
    :return: New background color after applying EMA.
    """
    bg_color = np.zeros_like(new)
    for i in range(3):  # RGB channels only (skip alpha)
        if new[i] > prev[i]:
            bg_color[i] = alpha_up * new[i] + (1 - alpha_up) * prev[i]
        else:
            bg_color[i] = alpha_down * new[i] + (1 - alpha_down) * prev[i]
    bg_color[3] = 1.0  # Keep alpha at 1.0
    return bg_color