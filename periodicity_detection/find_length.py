import numpy as np
from scipy.signal import argrelextrema
from statsmodels.tsa.stattools import acf


def find_length(data: np.ndarray) -> int:
    """Determine period size based on ACF.

    This method of determining the period size of a signal uses the highest spike in
    the ACF. The idea is taken from
    https://github.com/TheDatumOrg/TSB-UAD/blob/b61e91efc80be2d1766cc44f6a121c1b662ba67e/TSB_AD/utils/slidingWindows.py#L9.  # noqa: E501
    """
    base = 3
    nlags = min(data.shape[0], max(data.shape[0] // 4, 400))
    auto_corr = acf(data, nlags=nlags, fft=True)[base:]

    local_max = argrelextrema(auto_corr, np.greater)[0]
    if local_max.shape[0] < 1:
        return 1
    max_local_max = np.argmax([auto_corr[lcm] for lcm in local_max])
    if local_max[max_local_max] < 3 or local_max[max_local_max] > nlags - 100:
        return 1
    return local_max[max_local_max] + base
