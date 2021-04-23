"""
Module: libfmp.c3.c3s1_audio_feature
Author: Frank Zalkow, Meinard Müller
License: The MIT license, https://opensource.org/licenses/MIT

This file is part of the FMP Notebooks (https://www.audiolabs-erlangen.de/FMP)
"""
import numpy as np
from numba import jit


@jit(nopython=True)
def f_pitch(p, pitch_ref=69, freq_ref=440.0):
    """Computes the center frequency/ies of a MIDI pitch

    Notebook: C3/C3S1_SpecLogFreq-Chromagram.ipynb

    Args:
        p (float): MIDI pitch value(s)
        pitch_ref (float): Reference pitch (default: 69)
        freq_ref (float): Frequency of reference pitch (default: 440.0)

    Returns:
        freqs (float): Frequency value(s)
    """
    return 2 ** ((p - pitch_ref) / 12) * freq_ref


@jit(nopython=True)
def pool_pitch(p, Fs, N, pitch_ref=69, freq_ref=440.0):
    """Computes the set of frequency indices that are assigned to a given pitch

    Notebook: C3/C3S1_SpecLogFreq-Chromagram.ipynb

    Args:
        p (float): MIDI pitch value
        Fs (scalar): Sampling rate
        N (int): Window size of Fourier fransform
        pitch_ref (float): Reference pitch (default: 69)
        freq_ref (float): Frequency of reference pitch (default: 440.0)

    Returns:
        k (np.ndarray): Set of frequency indices
    """
    lower = f_pitch(p - 0.5, pitch_ref, freq_ref)
    upper = f_pitch(p + 0.5, pitch_ref, freq_ref)
    k = np.arange(N // 2 + 1)
    k_freq = k * Fs / N  # F_coef(k, Fs, N)
    mask = np.logical_and(lower <= k_freq, k_freq < upper)
    return k[mask]


@jit(nopython=True)
def compute_spec_log_freq(Y, Fs, N):
    """Computes a log-frequency spectrogram

    Notebook: C3/C3S1_SpecLogFreq-Chromagram.ipynb

    Args:
        Y (np.ndarray): Magnitude or power spectrogram
        Fs (scalar): Sampling rate
        N (int): Window size of Fourier fransform

    Returns:
        Y_LF (np.ndarray): Log-frequency spectrogram
        F_coef_pitch (np.ndarray): Pitch values
    """
    Y_LF = np.zeros((128, Y.shape[1]))
    for p in range(128):
        k = pool_pitch(p, Fs, N)
        Y_LF[p, :] = Y[k, :].sum(axis=0)
    F_coef_pitch = np.arange(128)
    return Y_LF, F_coef_pitch


@jit(nopython=True)
def compute_chromagram(Y_LF):
    """Computes a chromagram

    Notebook: C3/C3S1_SpecLogFreq-Chromagram.ipynb

    Args:
        Y_LF (np.ndarray): Log-frequency spectrogram

    Returns:
        C (np.ndarray): Chromagram
    """
    C = np.zeros((12, Y_LF.shape[1]))
    p = np.arange(128)
    for c in range(12):
        mask = (p % 12) == c
        C[c, :] = Y_LF[mask, :].sum(axis=0)
    return C


def note_name(p):
    """Returns note name of pitch

    Notebook: C3/C3S1_SpecLogFreq-Chromagram.ipynb

    Args:
        p (int): Pitch value

    Returns:
        name (str): Note name
    """
    chroma = ['A', 'A$^\\sharp$', 'B', 'C', 'C$^\\sharp$', 'D', 'D$^\\sharp$', 'E', 'F', 'F$^\\sharp$', 'G',
              'G$^\\sharp$']
    name = chroma[(p - 69) % 12] + str(p // 12 - 1)
    return name
