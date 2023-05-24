"""
Module: libfmp.c6.c6s3_beat_tracking
Author: Meinard Müller
License: The MIT license, https://opensource.org/licenses/MIT

This file is part of the FMP Notebooks (https://www.audiolabs-erlangen.de/FMP)
"""


import numpy as np
import librosa
from matplotlib import pyplot as plt
import IPython.display as ipd

import libfmp.b


def compute_penalty(N, beat_ref):
    """| Compute penalty funtion used for beat tracking [FMP, Section 6.3.2]
    | Note: Concatenation of '0' because of Python indexing conventions

    Notebook: C6/C6S3_BeatTracking.ipynb

    Args:
        N (int): Length of vector representing penalty function
        beat_ref (int): Reference beat period (given in samples)

    Returns:
        penalty (np.ndarray): Penalty function
    """
    t = np.arange(1, N) / beat_ref
    penalty = -np.square(np.log2(t))
    t = np.concatenate((np.array([0]), t))
    penalty = np.concatenate((np.array([0]), penalty))
    return penalty


def compute_beat_sequence(novelty, beat_ref, penalty=None, factor=1.0, return_all=False):
    """| Compute beat sequence using dynamic programming [FMP, Section 6.3.2]
    | Note: Concatenation of '0' because of Python indexing conventions

    Notebook: C6/C6S3_BeatTracking.ipynb

    Args:
        novelty (np.ndarray): Novelty function
        beat_ref (int): Reference beat period
        penalty (np.ndarray): Penalty function (Default value = None)
        factor (float): Weight parameter for adjusting the penalty (Default value = 1.0)
        return_all (bool): Return details (Default value = False)

    Returns:
        B (np.ndarray): Optimal beat sequence
        D (np.ndarray): Accumulated score
        P (np.ndarray): Maximization information
    """
    N = len(novelty)
    if penalty is None:
        penalty = compute_penalty(N, beat_ref)
    penalty = penalty * factor
    novelty = np.concatenate((np.array([0]), novelty))
    D = np.zeros(N+1)
    P = np.zeros(N+1, dtype=int)
    D[1] = novelty[1]
    P[1] = 0
    # forward calculation
    for n in range(2, N+1):
        m_indices = np.arange(1, n)
        scores = D[m_indices] + penalty[n-m_indices]
        maxium = np.max(scores)
        if maxium <= 0:
            D[n] = novelty[n]
            P[n] = 0
        else:
            D[n] = novelty[n] + maxium
            P[n] = np.argmax(scores) + 1
    # backtracking
    B = np.zeros(N, dtype=int)
    k = 0
    B[k] = np.argmax(D)
    while P[B[k]] != 0:
        k = k+1
        B[k] = P[B[k-1]]
    B = B[0:k+1]
    B = B[::-1]
    B = B - 1
    if return_all:
        return B, D, P
    else:
        return B


def beat_period_to_tempo(beat, Fs):
    """Convert beat period (samples) to tempo (BPM) [FMP, Section 6.3.2]

    Notebook: C6/C6S3_BeatTracking.ipynb

    Args:
        beat (int): Beat period (samples)
        Fs (scalar): Sample rate

    Returns:
        tempo (float): Tempo (BPM)
    """
    tempo = 60 / (beat / Fs)
    return tempo


def compute_plot_sonify_beat(x, Fs, nov, Fs_nov, beat_ref, factor, title=None, figsize=(6, 2)):
    """Compute, plot, and sonify beat sequence from novelty function [FMP, Section 6.3.2]

    Notebook: C6/C6S3_BeatTracking.ipynb

    Args:
        x: Novelty function
        Fs: Sample rate
        nov: Novelty function
        Fs_nov: Rate of novelty function
        beat_ref: Reference beat period
        factor: Weight parameter for adjusting the penalty
        title: Title of figure (Default value = None)
        figsize: Size of figure (Default value = (6, 2))
    """
    B = compute_beat_sequence(nov, beat_ref=beat_ref, factor=factor)

    beats = np.zeros(len(nov))
    beats[np.array(B, dtype=np.int32)] = 1
    if title is None:
        tempo = beat_period_to_tempo(beat_ref, Fs_nov)
        title = (r'Optimal beat sequence ($\hat{\delta}=%d$, $F_\mathrm{s}=%d$, '
                 r'$\hat{\tau}=%0.0f$ BPM, $\lambda=%0.2f$)' % (beat_ref, Fs_nov, tempo, factor))

    fig, ax, line = libfmp.b.plot_signal(nov, Fs_nov, color='k', title=title, figsize=figsize)
    T_coef = np.arange(nov.shape[0]) / Fs_nov
    ax.plot(T_coef, beats, ':r', linewidth=1)
    plt.show()

    beats_sec = T_coef[B]
    x_peaks = librosa.clicks(beats_sec, sr=Fs, click_freq=1000, length=len(x))
    ipd.display(ipd.Audio(x + x_peaks, rate=Fs))
