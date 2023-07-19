import numpy as np


def fft(data: np.ndarray) -> int:
    """Estimate the period of a time series using Fast Fourier Transform (FFT).

    This method computes the FFT of the time series and returns the index of the
    largest peak. This peak corresponds to the frequency :math:`f`, so the frequency is
    converted to a period using :math:`ceil(1 / f)`. If no peak is found, the period is
    estimated as 1.

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
    >>> from periodicity_detection import fft
    >>> period = fft(data)

    References
    ----------
    `<https://stackoverflow.com/a/59267175>`_ :
        StackOverflow answer on which this method is based on.
    """
    ft = np.fft.rfft(data)
    freqs = np.fft.rfftfreq(len(data))  # Get frequency axis from the time axis
    mags = abs(ft)  # We don't care about the phase information here
    inflection = np.diff(np.sign(np.diff(mags)))
    peaks = (inflection < 0).nonzero()[0] + 1
    peak = peaks[mags[peaks].argmax()]
    signal_freq = freqs[peak]
    period = int(1 // signal_freq)
    return period
