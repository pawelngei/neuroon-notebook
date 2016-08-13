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
  samples_per_second=1000, time_shift=0
):
  correlation_array = np.zeros(max_shift)
  for i in range(max_shift):
    correlation = np.correlate(
      first_signal[
        (i + time_shift) * samples_per_second :
        (i + time_shift + window_length) * samples_per_second
      ],
      second_signal[
        time_shift * samples_per_second :
        (time_shift + window_length) * samples_per_second
      ]
    )
    correlation_array[i] = correlation
  return correlation_array

def find_max_correlation(
  first_signal, second_signal,
  signal_start=timedelta(hours=1),
  window_shift=timedelta(hours=1), # how many steps to make
  window_length=timedelta(hours=1), # how long is the compared fragment
  frequency=1000
):
  '''
  Computes correlation between two signals.
  Returns a tuple of correlation array and max point.
  '''
  sample_length = window_shift + window_length
  first_window_length = np.logical_and(
    first_signal['timestamp'] > first_signal['timestamp'][0] + signal_start,
    first_signal['timestamp'] <= first_signal['timestamp'][0] + signal_start + sample_length
  )
  first_sample = first_signal[first_window_length]
  second_signal_length = np.logical_and(
    second_signal['timestamp'] > second_signal['timestamp'][0] + signal_start,
    second_signal['timestamp'] <= second_signal['timestamp'][0] + signal_start + sample_length
  )
  second_sample = second_signal[second_signal_length]
  upsample_length = sample_length.seconds * frequency
  first_upsampled_sample = signal.resample(first_sample.signal, upsample_length)
  second_upsampled_sample = signal.resample(second_sample.signal, upsample_length)
  correlated_array = correlate_with_time_shift(
    first_upsampled_sample,
    second_upsampled_sample
  )
  correlation_max = np.argmax(correlated_array)
  return (correlated_array, correlation_max)
