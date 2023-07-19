import numpy as np
from scipy.signal import argrelextrema
from statsmodels.tsa.stattools import acf

ACF_BASE = 3
MAX_LAGS = 400


def find_length(data: np.ndarray) -> int:
    """`find_length`-method from the TSB-UAD repository.

    This method of determining the period size of a signal uses the highest spike in
    the ACF. The idea is taken from the `TSB-UAD repository
    <https://github.com/TheDatumOrg/TSB-UAD/blob/b61e91efc80be2d1766cc44f6a121c1b662ba67e/TSB_AD/utils/slidingWindows.py#L9>`_.  # noqa: E501

    .. note::
       This method uses a couple of magic numbers and might not work well on some
       datasets.

    Parameters
    ----------
    data : array_like
        Array containing the time series data.

    Returns
    -------
    period : int
        Estimated period size of the time series.

    Examples
    --------

    Estimate the period length of a simple sine curve:

    >>> import numpy as np
    >>> rng = np.random.default_rng(42)
    >>> data = np.sin(np.linspace(0, 8*np.pi, 1000)) + rng.random(1000)/10
    >>> from periodicity_detection import find_length
    >>> period = find_length(data)

    References
    ----------
    `<https://github.com/TheDatumOrg/TSB-UAD>`_ :
        TSB-UAD repository.
    """
    nlags = min(data.shape[0], max(data.shape[0] // 4, MAX_LAGS))
    auto_corr = acf(data, nlags=nlags, fft=True)[ACF_BASE:]

    local_max = argrelextrema(auto_corr, np.greater)[0]
    if local_max.shape[0] < 1:
        return 1
    max_local_max = np.argmax([auto_corr[lcm] for lcm in local_max])
    if local_max[max_local_max] < 3 or local_max[max_local_max] > nlags - 100:
        return 1
    return local_max[max_local_max] + ACF_BASE
