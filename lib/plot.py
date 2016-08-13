import numpy as np
from scipy import signal
import pandas as pd
import pyedflib as el
from datetime import datetime, timedelta
from matplotlib import pyplot as pp
import seaborn as sns
pp.rcParams['figure.figsize'] = (16.0, 8.0)

def plot_signal_fragment(
  csv_sig, edf_sig, seconds=10, shift=0,
  csv_title='NeuroOn signal',
  edf_title='Aura PSG signal'
):
  '''
  Simple function used to plot two signals, based on their timestamps.
  '''
  plotting_series = [
    [211, csv_title, csv_sig],
    [212, edf_title, edf_sig]
  ]
  for series in plotting_series:
    pp.subplot(series[0])
    pp.title(series[1])
    signal = series[2]
    time_slice = np.logical_and(
      signal['timestamp'] >= signal['timestamp'][0] + timedelta(seconds=shift),
      signal['timestamp'] < signal['timestamp'][0] + timedelta(seconds=shift + seconds)
    )
    x_axis = signal['timestamp'][time_slice]
    y_axis = signal['signal'][time_slice]
    pp.plot(x_axis, y_axis)
  return pp.show()

def plot_spectrum_fragment(
  csv_sig, edf_sig, seconds=10, shift=0,
  cap_frequency=False,
  csv_title='NeuroOn spectrum',
  edf_title='Aura PSG spectrum'
):
  '''
  Simple function used to plot spectrum of two signals, based on their timestamps.
  '''
  plotting_series = [
    [211, csv_title, csv_sig],
    [212, edf_title, edf_sig]
  ]
  for series in plotting_series:
    pp.subplot(series[0])
    pp.title(series[1])
    signal = series[2]
    signal_freq = len(signal[
      signal['timestamp'] < signal['timestamp'][0] + timedelta(seconds=1)
    ])
    time_slice = np.logical_and(
      signal['timestamp'] >= signal['timestamp'][0] + timedelta(seconds=shift),
      signal['timestamp'] < signal['timestamp'][0] + timedelta(seconds=shift + seconds)
    )
    signal_len = len(signal[time_slice])
    signal_y = signal[time_slice]['signal']
    spectrum_y = np.fft.fft(signal_y) / signal_len
    spectrum_x = np.arange(signal_len) / (signal_len / signal_freq)
    half_spectrum = int(signal_len / 2)
    x_axis = spectrum_x[:half_spectrum]
    y_axis = abs(spectrum_y[:half_spectrum])
    if (cap_frequency):
      pp.xlim(0, cap_frequency)
    pp.plot(x_axis, y_axis)
  pp.show()


def plot_spectrum_fragment(
  csv_sig, edf_sig,
  seconds=10,
  shift=0,
  initial_timestamp=False,
  cap_frequency=False,
  csv_title='NeuroOn signal',
  csv_label='NeuroOn spectrum',
  edf_title='Aura PSG signal',
  edf_label='Aura PSG spectrum'
):
  '''
  Plot two signals on separate subplots and their spectrum on a third one.
  Based on timestamps.
  '''
  plotting_series = [
    {
      'subplot': 221,
      'title': csv_title,
      'label': csv_label,
      'signal': csv_sig,
      'color': sns.xkcd_rgb['denim blue']
    },
    {
      'subplot': 222,
      'title': edf_title,
      'label': edf_label,
      'signal': edf_sig,
      'color': sns.xkcd_rgb['medium green']
    }
  ]
  if not initial_timestamp:
    csv_init_ts = csv_sig.timestamp[0]
    edf_init_ts = edf_sig.timestamp[0]
    initial_timestamp = csv_init_ts if csv_init_ts > edf_init_ts else edf_init_ts
  for series in plotting_series:
    pp.subplot(series['subplot'])
    pp.title(series['title'])
    signal = series['signal']
    color = series['color']
    signal_freq = len(signal[
      signal['timestamp'] < signal['timestamp'][0] + timedelta(seconds=1)
    ])
    time_slice = np.logical_and(
      signal['timestamp'] >= initial_timestamp + timedelta(seconds=shift),
      signal['timestamp'] < initial_timestamp + timedelta(seconds=shift + seconds)
    )
    signal_len = len(signal[time_slice])
    signal_y = signal[time_slice]['signal']
    signal_x = signal[time_slice]['timestamp']
    pp.plot(signal_x, signal_y, color)
    pp.subplot(212)
    spectrum_y = np.fft.fft(signal_y) / signal_len
    spectrum_x = np.arange(signal_len) / (signal_len / signal_freq)
    half_spectrum = int(signal_len / 2)
    x_axis = spectrum_x[:half_spectrum]
    y_axis = abs(spectrum_y[:half_spectrum])
    if (cap_frequency):
      pp.xlim(0, cap_frequency)
    pp.plot(x_axis, y_axis, color, label=series['label'])
  pp.legend()
  pp.show()
