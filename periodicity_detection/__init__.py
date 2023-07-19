from typing import Optional, Any

import numpy as np

from ._version import __version__  # noqa

period_estimation_methods = [
    "find_length",
    "number_peaks",
    "autocorrelation",
    "fft",
    "autoperiod",
    "findfrequency",
]


def estimate_periodicity(
    data: np.ndarray,
    method: str = "find_length",
    use_initial_n: Optional[int] = None,
    **kwargs: Any,
) -> int:
    if len(data.shape) > 1:
        raise ValueError(
            "Cannot determine period length of multivariate time series; "
            "required shape is (n,)"
        )

    if use_initial_n is not None:
        n = min(use_initial_n, data.shape[0])
        data = data[:n]

    if method == "find_length":
        from .find_length import find_length

        return find_length(data)

    elif method == "number_peaks":
        from .number_peaks import number_peaks

        n = kwargs.get("n", 100)
        return number_peaks(data, n=n)

    elif method == "autocorrelation":
        from .autocorrelation import autocorrelation

        return autocorrelation(data)

    elif method == "autoperiod":
        from .autoperiod import Autoperiod

        pt_n_iter = kwargs.get("pt_n_iter", 100)
        random_state = kwargs.get("random_state", 42)
        plot = kwargs.get("plot", False)
        verbose = kwargs.get("verbose", 0)
        detrend = kwargs.get("detrend", True)
        use_number_peaks_fallback = kwargs.get("use_number_peaks_fallback", True)
        n = kwargs.get("n", 100)
        return Autoperiod(
            pt_n_iter=pt_n_iter,
            random_state=random_state,
            plot=plot,
            verbose=verbose,
            detrend=detrend,
            use_number_peaks_fallback=use_number_peaks_fallback,
            number_peaks_n=n,
        )(data)

    elif method == "fft":
        from .fft import fft

        return fft(data)

    elif method == "findfrequency":
        from .findfrequency import FindFrequency

        plot = kwargs.get("plot", False)
        verbose = kwargs.get("verbose", 0)
        detrend = kwargs.get("detrend", True)
        return FindFrequency(plot=plot, verbose=verbose, detrend=detrend)(data)

    else:
        return 1
