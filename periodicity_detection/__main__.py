import argparse
import sys
from pathlib import Path

import numpy as np

from ._version import __version__
from .methods import autocorrelation
from .methods import fft
from .methods import find_length
from .methods import number_peaks
from .methods.autoperiod import Autoperiod
from .methods.findfrequency import FindFrequency


def register_periodicity_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "dataset_path",
        type=Path,
        help="Path to the dataset for which the dominant period size should "
        "be estimated.",
    )
    parser.add_argument(
        "--use-initial-n",
        type=int,
        help="Only use the n initial points of the dataset to calculate the estimated "
        "period size.",
    )
    parser.add_argument(
        "--channel",
        type=int,
        help="If the dataset is multivariate, use the channel on this integer "
        "position. The first dimension is always assumed to be the index and "
        "skipped over!",
    )

    # subsubparsers for the individual tasks
    subparsers = parser.add_subparsers(dest="subcommand", required=True)

    # find length algorithm
    subparsers.add_parser(
        "find-length",
        help="Determine period size based on ACF as in the TSB-UAD repository.",
    )

    # number peaks algorithm
    method_parser = subparsers.add_parser(
        "number-peaks",
        help="Calculates the number of peaks of at least support n in the time "
        "series and the time series length divided by the number of peaks "
        "defines the period size.",
    )
    method_parser.add_argument(
        "--min-support",
        type=int,
        default=100,
        help="A peak of support n is defined as a subsequence where a value occurs, "
        "which is bigger than its n neighbours to the left and to the right.",
    )

    # autocorrelation algorithm
    subparsers.add_parser("autocorrelation", help="Determine period size based on ACF.")

    # fft algorithm
    subparsers.add_parser("fft", help="Determine period size based on FFT.")

    # autoperiod algorithm
    method_parser = subparsers.add_parser(
        "autoperiod",
        help="AUTOPERIOD method calculates the period size in a two-step process. "
        "First, it extracts candidate periods from the periodogram. Then, it "
        "uses the circular autocorrelation to validate the candidate periods.",
    )
    method_parser.add_argument(
        "--power-threshold-iter",
        type=int,
        default=100,
        help="Number of iterations to determine the power threshold.",
    )
    method_parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Seed for the random number generator.",
    )
    method_parser.add_argument(
        "--plot", action="store_true", help="Plot the periodogram and ACF."
    )
    method_parser.add_argument(
        "--verbose",
        type=int,
        default=0,
        help="Controls the log output verbosity. If set to 0, no messages are printed.",
    )
    method_parser.add_argument(
        "--disable-detrending",
        action="store_true",
        help="Don't remove potential linear trend from the time series before "
        "calculating periodogram and autocorrelation.",
    )
    method_parser.add_argument("--number-peaks-fallback", action="store_true", help="")
    method_parser.add_argument(
        "--min-support",
        type=int,
        default=100,
        help="Support for the number-peaks fallback method.",
    )

    # findfrequency algorithm
    method_parser = subparsers.add_parser(
        "findfrequency",
        help="Determine period size using the method findfrequency "
        "from the R forecast package. Re-implementation!",
    )
    method_parser.add_argument("--plot", action="store_true", help="")
    method_parser.add_argument(
        "--verbose", type=int, default=0, help="Controls the log output verbosity."
    )
    method_parser.add_argument(
        "--disable-detrending",
        action="store_true",
        help="Don't remove potential linear trend from the time series before "
        "calculating periodogram and autocorrelation.",
    )


def main(args: argparse.Namespace) -> int:
    command = args.subcommand

    # load data
    cols = 1 if args.channel is None else args.channel
    data = np.genfromtxt(
        args.dataset_path,
        delimiter=",",
        usecols=cols,
        skip_header=1,
        max_rows=args.use_initial_n,
    )

    if len(data.shape) > 1:
        raise ValueError(
            "Cannot determine period length of multivariate time series; required "
            f"shape is (n,), got {data.shape}. Please check your --channel and "
            "--use-initial-n parameters."
        )

    if command == "find-length":
        return find_length(data)

    elif command == "number-peaks":
        return number_peaks(data, n=args.min_support)

    elif command == "autocorrelation":
        return autocorrelation(data)

    elif command == "autoperiod":
        period = Autoperiod(
            pt_n_iter=args.power_threshold_iter,
            random_state=args.random_state,
            plot=args.plot,
            verbose=args.verbose,
            detrend=not args.disable_detrending,
            use_number_peaks_fallback=args.number_peaks_fallback,
            number_peaks_n=args.min_support,
            return_multi=1,
        )(data)
        if args.plot:
            import matplotlib.pyplot as plt

            plt.show()
        return period  # type: ignore

    elif command == "fft":
        return fft(data)

    elif command == "findfrequency":
        period = FindFrequency(
            plot=args.plot, verbose=args.verbose, detrend=not args.disable_detrending
        )(data)
        if args.plot:
            import matplotlib.pyplot as plt

            plt.show()
        return period

    else:
        raise ValueError(f"Unknown method for period estimation: {command}!")


def cli() -> None:
    parser = argparse.ArgumentParser(
        "periodicity",
        description="Detect the dominant period in univariate, equidistant time "
        "series data.",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"Periodicity v{__version__}",
        help="Show version number.",
    )
    register_periodicity_arguments(parser)

    argv = sys.argv[1:]
    args = parser.parse_args(argv)
    period = main(args)
    print(period)


if __name__ == "__main__":
    cli()
