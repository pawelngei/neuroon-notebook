import numpy as np
from scipy import signal
import pandas as pd
import pyedflib as el
from datetime import datetime, timedelta
from matplotlib import pyplot as pp
import seaborn as sns
pp.rcParams['figure.figsize'] = (16.0, 8.0)

def correlate_with_time_shift(
  first_signal, second_signal,
  max_shift=3600, window_length=3600,
  frequency=1000, time_shift=0
):
  correlation_array = np.zeros(max_shift)
  for i in range(max_shift):
    correlation = np.correlate(
      first_signal[
        (i + time_shift) * frequency :
        (i + time_shift + window_length) * frequency
      ],
      second_signal[
        time_shift * frequency :
        (time_shift + window_length) * frequency
      ]
    )
    correlation_array[i] = correlation
  return correlation_array
