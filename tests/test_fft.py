import numpy as np

from periodicity_detection.fft import fft


def test_detects_period():
    N = 1000
    data = np.sin(np.linspace(0, 40 * np.pi, N))
    # allow a deviation of 1 point:
    np.testing.assert_allclose(fft(data), 50, atol=1)
