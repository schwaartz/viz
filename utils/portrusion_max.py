import numpy as np

def softmax_protrusion_array(protrs: list[list[float]]) -> list[list[float]]:
    """
    Apply softmax to the protrusion array. The argument list is not modified.
    :param protrs: Protrusion array with size MAX_ACTIVE_PROTRUSIONS and elements [amount, size].
    :return: Protrusion array with softmax applied to the size.
    """
    # Convert to numpy array for easier manipulation
    protrs_np = np.array(protrs, dtype=np.float32)
    exp_values = np.exp(protrs_np[:, 1] - np.max(protrs_np[:, 1]))  # Subtract max for numerical stability
    softmax_values = exp_values / np.sum(exp_values)
    protrs_np[:, 1] = softmax_values
    return protrs_np.tolist()

def hardmax_protrusion_array(protrs: list[list[float]]) -> list[list[float]]:
    """
    Apply hardmax to the protrusion array. The argument list is not modified.
    :param protrs: Protrusion array with size MAX_ACTIVE_PROTRUSIONS and elements [amount, size].
    :return: Protrusion array with hardmax applied to the size.
    """
    protrs_np = np.array(protrs, dtype=np.float32)
    max_index = np.argmax(protrs_np[:, 1])
    protrs_np[:, 1] = 0.0
    protrs_np[max_index, 1] = 1.0
    return protrs_np.tolist()