"""
Module: libfmp.c2.c2_interference
Author: Meinard Müller
License: The MIT license, https://opensource.org/licenses/MIT

This file is part of the FMP Notebooks (https://www.audiolabs-erlangen.de/FMP)
"""
import numpy as np
from matplotlib import pyplot as plt


def plot_interference(x1, x2, t, figsize=(8, 2), xlim=None, ylim=None, title=''):
    """Helper function for plotting two signals and its superposition

    Notebook: C2/C2S3_InterferenceBeating.ipynb

    Args:
        x1: Signal 1
        x2: Signal 2
        t: Time
        figsize: figure size (Default value = (8, 2))
        xlim: x limits (Default value = None)
        ylim: y limits (Default value = None)
        title: figure title (Default value = '')
    """
    plt.figure(figsize=figsize)
    plt.plot(t, x1, color='gray', linewidth=1.0, linestyle='-', label='x1')
    plt.plot(t, x2, color='cyan', linewidth=1.0, linestyle='-', label='x2')
    plt.plot(t, x1+x2, color='red', linewidth=2.0, linestyle='-', label='x1+x2')
    if xlim is None:
        plt.xlim([0, t[-1]])
    else:
        plt.xlim(xlim)
    if ylim is not None:
        plt.ylim(ylim)
    plt.xlabel('Time (seconds)')
    plt.ylabel('Amplitude')
    plt.title(title)
    plt.legend(loc='upper right')
    plt.tight_layout()
    plt.show()


def generate_chirp_linear(dur, freq_start, freq_end, amp=1.0, Fs=22050):
    """Generation chirp with linear frequency increase

    Notebook: C2/C2S3_InterferenceBeating.ipynb

    Args:
        dur (float): Duration (seconds) of the signal
        freq_start (float): Start frequency of the chirp
        freq_end (float): End frequency of the chirp
        amp (float): Amplitude of chirp (Default value = 1.0)
        Fs (scalar): Sampling rate (Default value = 22050)

    Returns:
        x (np.ndarray): Generated chirp signal
        t (np.ndarray): Time axis (in seconds)
        freq (np.ndarray): Instant frequency (in Hz)
    """
    N = int(dur * Fs)
    t = np.arange(N) / Fs
    a = (freq_end - freq_start) / dur
    freq = a * t + freq_start
    x = amp * np.sin(np.pi * a * t ** 2 + 2 * np.pi * freq_start * t)
    return x, t, freq
