import numpy as np


def autocorrelation(data: np.ndarray) -> int:
    """Estimate the period of a time series using autocorrelation.

    This method computes the autocorrelation of the time series and returns the
    index of the largest peak. If no peak is found, the period is estimated as 1.

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
    >>> from periodicity_detection import autocorrelation
    >>> period = autocorrelation(data)

    References
    ----------
    `<https://stackoverflow.com/a/59267175>`_ :
        StackOverflow answer on which this method is based on.
    """
    acf = np.correlate(data, data, "full")[-len(data) :]

    # Find the second-order differences
    inflection = np.diff(np.sign(np.diff(acf)))

    # Find where they are negative
    peaks = (inflection < 0).nonzero()[0] + 1
    if peaks.shape[0] < 1:
        return 1

    # Of those, find the index with the maximum value
    period = peaks[acf[peaks].argmax()]
    return period
