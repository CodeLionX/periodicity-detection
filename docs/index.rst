periodicity-detection
=====================

.. toctree::
   :maxdepth: 1
   :hidden:

   user/index
   api/index

Overview
--------

tbd

.. code-block:: python

   import numpy as np
   import periodicity_detection as pyd

   # Create sample data
   data = np.sin(np.linspace(0, 40 * np.pi, 1000)) + np.random.default_rng(42).random(1000)

   # Calculate period size using a specific method
   period_size = pyd.findfrequency(data, detrend=True)
   assert period_size == 50

   # Calculate period size using the default method
   period_size = pyd.estimate_periodicity(data)
   assert period_size == 50

Supported period estimation algorithms
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

- :func:`~periodicity_detection.autocorrelation` reference: `<https://stackoverflow.com/a/59267175>`_.
- :func:`~periodicity_detection.autoperiod`, reference: `<https://epubs.siam.org/doi/epdf/10.1137/1.9781611972757.40>`_.
- :func:`~periodicity_detection.fft`, Fast Fourier Transform (FFT)-based method, reference:
  `<https://stackoverflow.com/a/59267175>`_.
- :func:`~periodicity_detection.find_length` from the `TSB-UAD <https://github.com/TheDatumOrg/TSB-UAD>`_ repository.
- :func:`~periodicity_detection.findfrequency`, Python-adaption of the R package ``forecast``'s
  ``findfrequency``-`function <https://rdrr.io/cran/forecast/man/findfrequency.html>`_
- :func:`~periodicity_detection.number_peaks`, Number of Peaks-method from tsfresh, source:
  :func:`~tsfresh.feature_extraction.feature_calculators.number_peaks`.

Installation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Prerequisites:

- python >= 3.7, <= 3.11
- pip >= 20

periodicity-detection is published to `PyPI <https://pypi.org/project/periodicity-detection>`_ and you can install it
using `pip`:

.. code-block:: shell

   pip install periodicity-detection

License
^^^^^^^

The project is licensed under the `MIT license <https://mit-license.org>`_. See the
`LICENSE <https://github.com/CodeLionX/periodicity-detection/blob/main/LICENSE>`_ file for more details.

    Copyright (c) 2022-2023 Sebastian Schmidl

User Guide
----------

Check out our :doc:`/user/index` to get started with the periodicity-detection package.
The user guides explain the API and the CLI.

:doc:`To the user guide</user/index>`

API Reference
-------------

The API reference guide contains a detailed description of the functions, modules, and objects included in this package.
The API reference describes how the methods work and which parameters can be used.

:doc:`To the API reference</api/index>`
