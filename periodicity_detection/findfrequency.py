import sys
from typing import Tuple

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.stats import linregress
from spectrum import aryule, arma2psd


class FindFrequency:
    """
    See Also
    --------

    `https://rdrr.io/cran/forecast/man/findfrequency.html`_ :
        original implementation in R
    https://rdrr.io/r/stats/spec.ar.html :
        ``spec.ar`` is used in the original implementation to compute the time series
        spectrum (PSD)
    https://rdrr.io/r/stats/ar.html :
        ``spec.ar`` uses this implementation to estimate the order and fit the AR model
        to the time series
    https://pyspectrum.readthedocs.io/en/latest/ref_param.html#spectrum.yulewalker.pyule :  # noqa: E501
        yule walker Python implementation used in this function
    """

    def __init__(self, plot: bool = False, verbose: int = 0, detrend: bool = True):
        self._plot = plot
        self._verbosity = verbose
        self.N_freq = 500
        self._detrend = detrend

    def _print(self, msg: str, level: int = 1) -> None:
        if self._verbosity >= level:
            print("  " * (level - 1) + msg, file=sys.stderr)

    def _ar_yule_walker(self, x: np.ndarray) -> Tuple[np.ndarray, float, int]:
        N = x.shape[0]

        # calculate max_order
        max_order = int(min(N - 1, np.floor(10 * np.log10(N))))
        if max_order < 1:
            raise ValueError("Maximum order must be >= 1!")
        self._print(f"order in range 0 to {max_order}", level=2)

        # demean by subtracting mean
        x = x - x.mean()

        # fit AR models for all orders
        ar_coeffs = []
        ar_vars = []
        orders = np.arange(0, max_order + 1)  # include max_order as last element
        for order in orders:
            # unfortunately, statsmodels yule_walker method is not robust enough for
            # our purposes
            # from statsmodels.regression import yule_walker
            # ar, var = yule_walker(x, order=order, demean=False)
            ar, var, _ = aryule(x, order=order)
            ar_coeffs.append(ar)
            ar_vars.append(var)

        # calculate AICs
        # formula directly taken from R stats-library:
        # https://github.com/wch/r-source/blob/79298c499218846d14500255efd622b5021c10ec/src/library/stats/R/ar.R#L141  # noqa: E501
        aics = N * np.log(ar_vars) + 2 * orders + 2
        aics -= min(aics)

        # select order with minimum AIC
        k = np.argmin(aics)
        ar = ar_coeffs[k]
        var = ar_vars[k]
        order = orders[k]
        self._print(f"selected order with smallest AIC: {order}", level=2)
        self._print(f"AR coefficients: {ar}", level=2)
        self._print(f"AR variance: {var}", level=2)

        if self._plot:
            fig, axs = plt.subplots(1, 1, squeeze=False)
            axs[0, 0].set_title("AIC vs. order")
            axs[0, 0].plot(orders, aics, label="AIC", color="blue")
            axs[0, 0].vlines(
                order,
                aics.min(),
                aics.max(),
                label=f"Minimum AIC (order={order})",
                color="orange",
            )
            axs[0, 0].set_xlabel("order")
            axs[0, 0].set_ylabel("AIC")
            axs[0, 0].legend()
        return ar, var, order

    # TODO: optimize (precomputed LUT for cs and sn?)
    def _spectrum(self, ar: np.ndarray, var: float, order: int) -> np.ndarray:
        """Compute spectrum from AR coefficients and variance.

        Algorithm directly taken from R stats-library:
        https://github.com/wch/r-source/blob/79298c499218846d14500255efd622b5021c10ec/src/library/stats/R/spectrum.R#L76  # noqa: E501
        """
        if order >= 1:
            freq = np.linspace(0, 0.5, self.N_freq)
            cs = np.empty((self.N_freq, order))
            sn = np.empty((self.N_freq, order))
            for i, f in enumerate(freq):
                for j, o in enumerate(range(1, order + 1)):
                    cs[i, j] = np.cos(2 * np.pi * f * o)
                    sn[i, j] = np.sin(2 * np.pi * f * o)
            cs = np.matmul(cs, ar)
            sn = np.matmul(sn, ar)
            spec = var / ((1 - cs) ** 2 + sn**2)
        else:
            self._print("Spectrum is constant because order is 0", level=2)
            spec = np.repeat(var, self.N_freq)
        return spec

    def _find_peak_fap(self, psd: np.ndarray) -> Tuple[float, int]:
        """Find peak in spectrum (PSD) and computes its corresponding frequency and
        period.

        Algorithm directly taken from R forecast-library:
        https://rdrr.io/cran/forecast/src/R/findfrequency.R
        """
        const = 1e-4  # adjusted from R code (slightly different AR method)
        f = 0
        p = 1
        freq = np.linspace(0, 0.5, self.N_freq)
        if np.max(psd) > const:
            f = freq[np.argmax(psd)]
            p = np.floor(1 / f + 0.5)
            if np.isinf(p):
                self._print(
                    f"Peak at f={f} is Inf, searching next peak in spectrum", level=2
                )
                # find next local maximum
                j = np.argwhere(np.diff(psd) > 0).ravel()
                if j.shape[0] > 0:
                    i = 0
                    while np.isinf(p):
                        nextmax = j[i] + np.argmax(psd[j[i] + 1 :])
                        if nextmax < self.N_freq:
                            f = freq[nextmax]
                            p = np.floor(1 / f + 0.5)
                        else:
                            self._print(
                                "Peak search hit the end of the spectrum, no dominant "
                                "frequency/period found!"
                            )
                            p = 1
                        i += 1
                else:
                    self._print("No dominant frequency/period found!")
        if np.isinf(p):
            p = 1
        return f, int(p)

    def __call__(self, data: np.ndarray) -> int:
        index = np.arange(data.shape[0])
        if self._detrend:
            self._print("Detrending time series with linear model")
            trend_fit = linregress(index, data)
            self._print(f"fit stderr={trend_fit.stderr}", level=2)
            if trend_fit.slope > 1e-4:
                trend = trend_fit.intercept + index * trend_fit.slope
                data = data - trend
                self._print(
                    f"removed trend with slope {trend_fit.slope:.6f} and "
                    f"intercept {trend_fit.intercept:.4f}",
                    level=2,
                )
            else:
                self._print(
                    f"skipping detrending because slope ({trend_fit.slope:.6f}) "
                    "is to shallow (< 1e-4)",
                    level=2,
                )
                self._print(f"removing remaining mean ({data.mean():.4f})", level=2)
                data = data - data.mean()

        # check for NaNs
        if np.any(np.isnan(data)):
            raise ValueError("NaNs encountered in data!")
            # TODO: Use longest contiguous subsequence of non-NaNs

        self._print("Fitting AR models using yule walker method")
        ar, var, order = self._ar_yule_walker(data)

        self._print("Computing power spectrum")
        psd = arma2psd(ar, NFFT=self.N_freq * 2)[: self.N_freq]
        # psd = self._spectrum(ar, var, order=order)

        self._print("Detecting peak in power spectrum")
        f, p = self._find_peak_fap(psd)
        self._print(f"Frequency={f}, period={p}")

        if self._plot:
            fig, axs = plt.subplots(1, 1, squeeze=False)
            axs[0, 0].set_title("PSD")
            axs[0, 0].set_yscale("log")
            # axs[0, 0].plot(
            #     np.linspace(-0.5, 0.5, psd.shape[0]),
            #     10 * np.log10(psd/np.max(psd))
            # )
            freq = np.linspace(0, 0.5, self.N_freq)
            axs[0, 0].plot(freq, psd, color="blue", label="Spectrum")
            axs[0, 0].plot(
                f,
                psd[np.nonzero(freq == f)],
                "*",
                color="orange",
                label=f"Found period (f={f:.4f}, p={p})",
            )
            axs[0, 0].set_xlabel("frequency")
            axs[0, 0].set_ylabel("power")
            axs[0, 0].legend()
        return p


if __name__ == "__main__":
    df = pd.read_csv("data/tmp/taylor.csv", index_col=0)
    df["res"] = pd.read_csv("data/tmp/residuals.csv", index_col=0).iloc[:, 0]
    # df.plot(subplots=True)
    data = df.iloc[:, 0].values
    period = FindFrequency(plot=True, verbose=3)(data)

    print("findfrequency period =", period)

    # plt.figure()
    # data = pd.read_csv("data/tmp/spec.csv", index_col=0).iloc[:, 0]
    # plt.plot(data)
    # plt.yscale("log")
    plt.show()
