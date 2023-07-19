import numpy as np


def autocorrelation(data: np.ndarray) -> int:
    """https://stackoverflow.com/a/59267175"""
    acf = np.correlate(data, data, "full")[-len(data) :]
    inflection = np.diff(np.sign(np.diff(acf)))  # Find the second-order differences
    peaks = (inflection < 0).nonzero()[0] + 1  # Find where they are negative
    if peaks.shape[0] < 1:
        return 1
    period = peaks[
        acf[peaks].argmax()
    ]  # Of those, find the index with the maximum value
    return period
