from typing import Optional

import numpy as np


def number_peaks(data: np.ndarray, n: int) -> int:
    """Calculates the number of peaks of at least support n in the time series. A peak
    of support n is defined as a subsequence where a value occurs, which is bigger than
    its n neighbours to the left and to the right. The time series length divided by
    the number of peaks defines the period size.

    Parameters
    ----------
    data : array_like
        time series to calculate the number of peaks of
    n : int
        the required support for the peaks

    Returns
    -------
    period_size : float
        the estimated period size

    See Also
    --------
    tsfresh.feature_extraction.number_peaks :
        tsfresh's implementation, on which this method is based on
    """
    x_reduced = data[n:-n]

    res: Optional[np.ndarray] = None
    for i in range(1, n + 1):
        result_first = x_reduced > _roll(data, i)[n:-n]

        if res is None:
            res = result_first
        else:
            res &= result_first

        res &= x_reduced > _roll(data, -i)[n:-n]
    n_peaks = np.sum(res)  # type: ignore
    if n_peaks < 1:
        return 1
    return data.shape[0] // n_peaks


def _roll(a: np.ndarray, shift: int) -> np.ndarray:
    """Exact copy of tsfresh's ``_roll``-implementation:
    https://github.com/blue-yonder/tsfresh/blob/611e04fb6f7b24f745b4421bbfb7e986b1ec0ba1/tsfresh/feature_extraction/feature_calculators.py#L49  # noqa: E501

    This roll is for 1D arrays and significantly faster than ``np.roll()``.

    Parameters
    ----------
    a : array_like
        input array
    shift : int
        the number of places by which elements are shifted

    Returns
    -------
    array : array_like
        shifted array with the same shape as the input array ``a``

    See Also
    --------
    https://github.com/blue-yonder/tsfresh/blob/611e04fb6f7b24f745b4421bbfb7e986b1ec0ba1/tsfresh/feature_extraction/feature_calculators.py#L49 :  # noqa: E501
        Implementation in tsfresh.
    """
    if not isinstance(a, np.ndarray):
        a = np.asarray(a)
    idx = shift % len(a)
    return np.concatenate([a[-idx:], a[:-idx]])
