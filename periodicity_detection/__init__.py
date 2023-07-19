from typing import Optional, Any

import numpy as np

from ._version import __version__  # noqa
from .methods import (
    autocorrelation,
    find_length,
    number_peaks,
    fft,
)
from .methods import autoperiod  # noqa: F401
from .methods import findfrequency  # noqa: F401

PERIOD_ESTIMATION_METHODS = [
    "find_length",
    "number_peaks",
    "autocorrelation",
    "fft",
    "autoperiod",
    "findfrequency",
]
"""List of available methods for period estimation."""


def estimate_periodicity(
    data: np.ndarray,
    method: str = "find_length",
    use_initial_n: Optional[int] = None,
    **kwargs: Any,
) -> int:
    """Estimate the periodicity of a time series using one of the provided methods.

    Parameters
    ----------
    data : np.ndarray
        Univariate ime series data with equidistant time steps.
    method : str, optional
        Method to use for period estimation, by default "find_length". See
        :data:`~periodicity_detection.PERIOD_ESTIMATION_METHODS` for a list of available
        methods.
    use_initial_n : int, optional
        Use only the first `use_initial_n` data points for period estimation. This can
        be useful for very long time series, where the period is expected to be constant
        over the length of the time series. Be default, the entire time series is used.
    **kwargs : Any
        Additional keyword arguments passed to the respective method.

    Returns
    -------
    period : int
        Estimated period length of the time series.

    Examples
    --------

    Estimate the periodicity of an example dataset using the `find_length` method:

    >>> import numpy as np
    >>> import periodicity_detection as pyd
    >>> rng = np.random.default_rng(42)
    >>> data = np.sin(np.linspace(0, 10 * np.pi, 1000)) + rng.normal(0, 0.1, 1000)
    >>> pyd.estimate_periodicity(data, method="find_length")
    199
    """
    if method not in PERIOD_ESTIMATION_METHODS:
        raise ValueError(f"Unknown method '{method}'!")

    if len(data.shape) > 1:
        raise ValueError(
            "Cannot determine period length of multivariate time series; "
            "required shape is (n,)"
        )

    if use_initial_n is not None:
        n = min(use_initial_n, data.shape[0])
        data = data[:n]

    if method == "find_length":
        return find_length(data)

    elif method == "number_peaks":
        n = kwargs.get("n", 100)
        return number_peaks(data, n=n)

    elif method == "autocorrelation":
        return autocorrelation(data)

    elif method == "autoperiod":
        from .methods.autoperiod import Autoperiod

        pt_n_iter = kwargs.get("pt_n_iter", 100)
        random_state = kwargs.get("random_state", 42)
        plot = kwargs.get("plot", False)
        verbose = kwargs.get("verbose", 0)
        detrend = kwargs.get("detrend", True)
        use_number_peaks_fallback = kwargs.get("use_number_peaks_fallback", True)
        n = kwargs.get("n", 100)
        return Autoperiod(  # type: ignore
            pt_n_iter=pt_n_iter,
            random_state=random_state,
            plot=plot,
            verbose=verbose,
            detrend=detrend,
            use_number_peaks_fallback=use_number_peaks_fallback,
            number_peaks_n=n,
            return_multi=1,
        )(data)

    elif method == "fft":
        return fft(data)

    elif method == "findfrequency":
        from .methods.findfrequency import FindFrequency

        plot = kwargs.get("plot", False)
        verbose = kwargs.get("verbose", 0)
        detrend = kwargs.get("detrend", True)
        return FindFrequency(plot=plot, verbose=verbose, detrend=detrend)(data)

    else:
        return 1
