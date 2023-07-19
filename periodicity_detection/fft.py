import numpy as np


def fft(data: np.ndarray) -> int:
    """https://stackoverflow.com/a/59267175"""
    ft = np.fft.rfft(data)
    freqs = np.fft.rfftfreq(len(data))  # Get frequency axis from the time axis
    mags = abs(ft)  # We don't care about the phase information here
    inflection = np.diff(np.sign(np.diff(mags)))
    peaks = (inflection < 0).nonzero()[0] + 1
    peak = peaks[mags[peaks].argmax()]
    signal_freq = freqs[peak]  # Gives 0.05
    period = int(1 // signal_freq)
    return period
