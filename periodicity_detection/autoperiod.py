from __future__ import annotations

import sys
import warnings
from typing import Any, List, Tuple, Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import statsmodels.tsa.stattools
from matplotlib.patches import Rectangle
from scipy.optimize import minimize_scalar
from scipy.signal import periodogram
from scipy.stats import linregress

from .number_peaks import number_peaks


class Autoperiod:
    """AUTOPERIOD method calculates the period in a two-step process. First, it
    extracts candidate periods from the periodogram (using an automatically
    determined power threshold, see ``pt_n_iter`` parameter). Then, it uses the circular
    autocorrelation to validate the candidate periods. Periods on a hill of the ACF
    with sufficient steepness are considered valid. The candidate period with the
    highest power is returned.

    Changes compared to the paper:

    - Potential detrending of the time series before estimating the period.
    - Potentially returns multiple detected periodicities.

    Parameters
    ----------
    pt_n_iter : int
        Number of shuffling iterations to determine the power threshold. The higher the
        number, the tighter the confidence interval. The percentile is calculated using
        :math:`percentile = 1 - 1 / pt_n_iter`.
    random_state : Any
        Seed for the random number generator. Used for determining the power threshold
        (data shuffling).
    plot : bool
        Show the periodogram and ACF plots.
    verbose : int
        Controls the log output verbosity. If set to ``0``, no messages are printed;
        when ``>=3``, all messages are printed.
    detrend : bool
        Removes linear trend from the time series before calculating the candidate
        periods. (Addition to original method).
    return_multi : int
        Maximum number of periods to return.

    Examples
    --------

    Estimate the period length of a simple sine curve:

    >>> import numpy as np
    >>> rng = np.random.default_rng(42)
    >>> data = np.sin(np.linspace(0, 8*np.pi, 1000)) + rng.random(1000)/10
    >>> period = Autoperiod(random_state=42, detrend=True)(data)

    Plot the periodogram and ACF while computing the period size:

    >>> import numpy as np
    >>> import matplotlib.pyplot as plt
    >>> rng = np.random.default_rng(42)
    >>> data = np.sin(np.linspace(0, 8*np.pi, 1000)) + rng.random(1000)/10
    >>> period = Autoperiod(random_state=42, detrend=True, plot=True)(data)
    >>> plt.show()

    See Also
    --------
    `https://epubs.siam.org/doi/epdf/10.1137/1.9781611972757.40`_ : Paper reference
    """

    # potential improvement:
    # https://link.springer.com/chapter/10.1007/978-3-030-39098-3_4
    def __init__(
        self,
        *,
        pt_n_iter: int = 100,
        random_state: Any = None,
        plot: bool = False,
        verbose: int = 0,
        detrend: bool = False,
        use_number_peaks_fallback: bool = False,
        number_peaks_n: int = 100,
        return_multi: int = 1,
    ):
        self._pt_n_iter = pt_n_iter
        self._rng: np.random.Generator = np.random.default_rng(random_state)
        self._plot = plot
        self._verbosity = verbose
        self._detrend = detrend
        self._use_np_fb = use_number_peaks_fallback
        self._np_n = number_peaks_n
        self._trend: Optional[np.ndarray] = None
        self._orig_data: Optional[np.ndarray] = None
        self._return_multi = return_multi

    def __call__(self, data: np.ndarray) -> Union[List[int], int]:
        if self._detrend:
            self._print("Detrending")
            index = np.arange(data.shape[0])
            trend_fit = linregress(index, data)
            if trend_fit.slope > 1e-4:
                trend = trend_fit.intercept + index * trend_fit.slope
                if self._plot:
                    self._trend = trend
                    self._orig_data = data
                data = data - trend
                self._print(
                    f"removed trend with slope {trend_fit.slope:.6f} "
                    f"and intercept {trend_fit.intercept:.4f}",
                    level=2,
                )
            else:
                self._print(
                    f"skipping detrending because slope ({trend_fit.slope:.6f}) "
                    f"is too shallow (< 1e-4)",
                    level=2,
                )
                self._print(f"removing remaining mean ({data.mean():.4f})", level=2)
                data = data - data.mean()

        if self._verbosity > 1:
            self._print("Determining power threshold")
        p_threshold = self._power_threshold(data)
        self._print(f"Power threshold: {p_threshold:.6f}")

        if self._verbosity > 1:
            self._print("\nDiscovering candidate periods (hints) from periodogram")
        period_hints = self._candidate_periods(data, p_threshold)
        self._print(f"{len(period_hints)} candidate periods (hints)")

        if self._verbosity > 1:
            self._print("\nVerifying hints using ACF")
        periods = self._verify(data, period_hints)

        if len(periods) < 1 or periods[0] <= 1 and self._use_np_fb:
            self._print(
                f"\nDetected invalid period ({periods}), "
                f"falling back to number_peaks method"
            )
            periods = [number_peaks(data, n=self._np_n)]
        self._print(f"Periods are {periods}")
        if self._return_multi > 1:
            return periods[: self._return_multi]
        else:
            return int(periods[0])

    def _print(self, msg: str, level: int = 1) -> None:
        if self._verbosity >= level:
            print("  " * (level - 1) + msg, file=sys.stderr)

    def _power_threshold(self, data: np.ndarray) -> float:
        n_iter = self._pt_n_iter
        percentile = 1 - 1 / n_iter
        self._print(
            f"determined confidence interval as {percentile} "
            f"(using {n_iter} iterations)",
            level=2,
        )
        max_powers = []
        values = data.copy()
        for i in range(n_iter):
            self._rng.shuffle(values)
            _, p_den = periodogram(values)
            max_powers.append(np.max(p_den))
        max_powers.sort()
        return max_powers[-1]

    def _candidate_periods(
        self, data: np.ndarray, p_threshold: float
    ) -> List[Tuple[int, float, float]]:
        N = data.shape[0]
        f, p_den = periodogram(data)
        # k are the DFT bin indices (see paper)
        k = np.array(f * N, dtype=np.int_)
        # print("k:", k)
        # print("frequency:", f)  # between 0 and 0.5
        # print("period:", N/k)

        self._print(
            f"inspecting periodogram between 2 and {N // 2} (frequencies 0 and 0.5)",
            level=2,
        )
        period_hints = []
        for i in np.arange(2, N // 2):
            if p_den[i] > p_threshold:
                period_hints.append((k[i], f[i], p_den[i]))
                self._print(
                    f"detected hint at bin k={k[i]} (f={f[i]:.4f}, "
                    f"power={p_den[i]:.2f})",
                    level=3,
                )

        # start with the highest power frequency:
        self._print("sorting hints by highest power first", level=2)
        period_hints = sorted(period_hints, key=lambda x: x[-1], reverse=True)

        if self._plot:
            plt.figure()
            plt.title("Periodogram")
            plt.semilogy(f, p_den, color="blue", label="PSD")
            plt.hlines(
                [p_threshold], f[0], f[-1], color="orange", label="power threshold"
            )
            plt.plot(
                [p[1] for p in period_hints],
                [p[2] for p in period_hints],
                "*",
                color="red",
                label="period hints",
            )
            plt.xlabel("frequency")
            plt.ylabel("PSD")
            plt.legend()
            # plt.show()

        return period_hints

    def _verify(
        self, data: np.ndarray, period_hints: List[Tuple[int, float, float]]
    ) -> List[int]:
        # produces wrong acf:
        # acf = fftconvolve(data, data[::-1], 'full')[data.shape[0]:]
        # acf = acf / np.max(acf)
        acf = statsmodels.tsa.stattools.acf(data, fft=True, nlags=data.shape[0])
        index = np.arange(acf.shape[0])
        N = data.shape[0]
        ranges = []

        warnings.filterwarnings(
            action="ignore",
            category=RuntimeWarning,
            message=r".*invalid value encountered.*",
        )
        for k, f, power in period_hints:
            # determine search interval
            begin = int((N / (k + 1) + N / k) / 2) - 1
            end = int((N / k + N / (k - 1)) / 2) + 1
            while end - begin < 4:
                if begin > 0:
                    begin -= 1
                if end < N - 1:
                    end += 1
            self._print(
                f"processing hint at {N//k}, k={k}: begin={begin}, end={end+1}", level=2
            )
            slopes = {}

            def two_segment(t: float, args: List[np.ndarray]) -> float:
                x, y = args
                t = int(np.round(t))
                slope1 = linregress(x[:t], y[:t])
                slope2 = linregress(x[t:], y[t:])
                slopes[t] = (slope1, slope2)
                error = np.sum(
                    np.abs(y[:t] - (slope1.intercept + slope1.slope * x[:t]))
                ) + np.sum(np.abs(y[t:] - (slope2.intercept + slope2.slope * x[t:])))
                return error

            # print("outer indices", begin, end+1)
            # print("inner indices", 0, end - begin + 1)
            # print("bounds", 2, end - begin - 2)
            res = minimize_scalar(
                two_segment,
                args=[index[begin : end + 1], acf[begin : end + 1]],
                method="bounded",
                bounds=(2, end - begin - 2),
                options={
                    "disp": 1 if self._verbosity > 2 else 0,
                    "xatol": 1e-8,
                    "maxiter": 500,
                },
            )
            if not res.success:
                # self._print(f"curve fitting failed ({res.message}) --> INVALID", l=3)
                # continue
                raise ValueError(
                    "Failed to find optimal midway-point for slope-fitting "
                    f"(hint: k={k}, f={f}, power={power})!"
                )

            t = int(np.round(res.x))
            optimal_t = begin + t
            slope = slopes[t]
            self._print(f"found optimal t: {optimal_t} (t={t})", level=3)
            # self._print(f"slopes={slopes[0].slope:.4f},{slopes[1].slope:.4f}", l=3)
            # change from paper: we require a certain hill size to prevent noise
            # influencing our results:
            if (
                slope[0].slope > 0 > slope[1].slope
            ):  # and np.abs(slopes[0].slope) + np.abs(slopes[1].slope) > 0.01:
                self._print("hill detected --> VALID", level=3)
                period = begin + np.argmax(acf[begin : end + 1])
                self._print(f"corrected period (from {N // k}): {period}", level=3)
                ranges.append((begin, end, optimal_t, period, slope))
                if self._return_multi <= 1:
                    break
            elif slope[0].slope < 0 < slope[1].slope:
                self._print("valley detected --> INVALID", level=3)
            # elif np.abs(slopes[0].slope) + np.abs(slopes[1].slope) <= 0.01:
            #     self._print(
            #         f"insufficient steepness ({np.abs(slopes[0].slope):.4f} and "
            #         f"{np.abs(slopes[1].slope):.4f}) --> INVALID",
            #         l=3
            #     )
            else:
                self._print("not a hill, but also not a valley --> INVALID", level=3)
            # period = begin + np.argmax(acf[begin:end + 1])
            # ranges.append((begin, end, optimal_t, period, slopes))

        warnings.filterwarnings(
            action="default",
            category=RuntimeWarning,
            message=r".*invalid value encountered.*",
        )

        if self._plot:
            n_plots = 3 if self._detrend and self._trend else 2
            fig, axs = plt.subplots(n_plots, 1, sharex="col")
            axs[0].set_title("Original time series")
            axs[0].set_xlabel("time")

            if self._trend and self._orig_data:
                axs[0].plot(self._orig_data, label="time series", color="blue")
                axs[0].plot(self._trend, label="linear trend", color="black")
                axs[1].set_title("Detrended time series")
                axs[1].plot(data, label="time series", color="black")
                axs[1].set_xlabel("time")
            else:
                axs[0].plot(data, label="time series", color="black")

            axs[-1].set_title("Circular autocorrelation")
            axs[-1].plot(acf, label="ACF", color="blue")
            data_min = acf.min()
            data_max = acf.max()
            for i, (b, e, t, period, (slope1, slope2)) in enumerate(ranges):
                axs[-1].add_patch(
                    Rectangle(
                        (b, data_min),
                        e - b,
                        data_max - data_min,
                        color="yellow",
                        alpha=0.5,
                    )
                )
                axs[-1].plot(period, acf[period], "*", color="red", label="period")
                axs[-1].plot(
                    index[b:t],
                    slope1.intercept + slope1.slope * index[b:t],
                    color="orange",
                    label="fitted slope1",
                )
                axs[-1].plot(
                    index[t : e + 1],
                    slope2.intercept + slope2.slope * index[t : e + 1],
                    color="orange",
                    label="fitted slope2",
                )

            axs[-1].set_xlabel("lag (period size)")
            axs[-1].set_ylabel("magnitude")
            axs[-1].legend()
            # plt.show()

        periods = list(x[3] for x in ranges)
        if len(periods) > 0:
            return np.unique(periods)[::-1][: self._return_multi].astype(int)
        else:
            return [1]
