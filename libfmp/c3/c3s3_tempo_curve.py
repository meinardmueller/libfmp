"""
Module: libfmp.c3.c3s3_tempo_curve
Author: Meinard Müller, Frank Zalkow
License: The MIT license, https://opensource.org/licenses/MIT

This file is part of the FMP Notebooks (https://www.audiolabs-erlangen.de/FMP)
"""
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.ticker import ScalarFormatter
import librosa
from scipy import signal
from scipy.interpolate import interp1d
import scipy.ndimage.filters

import libfmp.c3


def compute_score_chromagram(score, Fs_beat):
    """Compute chromagram from score representation

    Notebook: C3/C3S3_MusicAppTempoCurve.ipynb

    Args:
        score (list): Score representation
        Fs_beat (scalar): Sampling rate for beat axis

    Returns:
        X_score (np.ndarray): Chromagram representation X_score
        t_beat (np.ndarray): Time axis t_beat (given in beats)
    """
    score_beat_min = min(n[0] for n in score)
    score_beat_max = max(n[0] + n[1] for n in score)
    beat_res = 1.0 / Fs_beat
    t_beat = np.arange(score_beat_min, score_beat_max, beat_res)
    X_score = np.zeros((12, len(t_beat)))

    for start, duration, pitch, velocity, label in score:
        start_idx = int(round(start / beat_res))
        end_idx = int(round((start + duration) / beat_res))
        cur_chroma = int(round(pitch)) % 12
        X_score[cur_chroma, start_idx:end_idx] += velocity

    X_score = librosa.util.normalize(X_score, norm=2)
    return X_score, t_beat


def plot_measure(ax, measure_pos):
    """Plot measure positions

    Notebook: C3/C3S3_MusicAppTempoCurve.ipynb

    Args:
        ax (mpl.axes.Axes): Figure axis
        measure_pos (list or np.ndarray): Array containing measure positions
    """
    y_min, y_max = ax.get_ylim()
    ax.vlines(measure_pos, y_min, y_max, color='r')
    for m in range(len(measure_pos)):
        ax.text(measure_pos[m], y_max, '%s' % (m + 1),
                color='r', backgroundcolor='mistyrose',
                verticalalignment='top', horizontalalignment='left')


def compute_strict_alignment_path(P):
    """Compute strict alignment path from a warping path

    Notebook: C3/C3S3_MusicAppTempoCurve.ipynb

    Args:
        P (list or np.ndarray): Warping path

    Returns:
        P_mod (list or np.ndarray): Strict alignment path
    """
    # Initialize P_mod and enforce start boundary condition
    P_mod = np.zeros(P.shape)
    P_mod[0] = P[0]
    N, M = P[-1]
    # Go through all cells of P until reaching last row or column
    assert N > 1 and M > 1, 'Length of sequences must be longer than one.'
    i, j = 0, 0
    n1, m1 = P[i]
    while True:
        i += 1
        n2, m2 = P[i]
        if n2 == N or m2 == M:
            # If last row or column is reached, quit loop
            break
        if n2 > n1 and m2 > m1:
            # Strict monotonicity condition is fulfuilled
            j += 1
            P_mod[j] = n2, m2
            n1, m1 = n2, m2
    j += 1
    # Enforce end boundary condition
    P_mod[j] = P[-1]
    P_mod = P_mod[:j+1]
    return P_mod


def compute_strict_alignment_path_mask(P):
    """Compute strict alignment path from a warping path

    Notebook: C3/C3S3_MusicAppTempoCurve.ipynb

    Args:
        P (list or np.ndarray): Wapring path

    Returns:
        P_mod (list or np.ndarray): Strict alignment path
    """
    P = np.array(P, copy=True)
    N, M = P[-1]
    # Get indices for strict monotonicity
    keep_mask = (P[1:, 0] > P[:-1, 0]) & (P[1:, 1] > P[:-1, 1])
    # Add first index to enforce start boundary condition
    keep_mask = np.concatenate(([True], keep_mask))
    # Remove all indices for of last row or column
    keep_mask[(P[:, 0] == N) | (P[:, 1] == M)] = False
    # Add last index to enforce end boundary condition
    keep_mask[-1] = True
    P_mod = P[keep_mask, :]

    return P_mod


def plot_tempo_curve(f_tempo, t_beat, ax=None, figsize=(8, 2), color='k', logscale=False,
                     xlabel='Time (beats)', ylabel='Temp (BPM)', xlim=None, ylim=None,
                     label='', measure_pos=[]):
    """Plot a tempo curve

    Notebook: C3/C3S3_MusicAppTempoCurve.ipynb

    Args:
        f_tempo: Tempo curve
        t_beat: Time axis of tempo curve (given as sampled beat axis)
        ax: Plot either as figure (ax==None) or into axis (ax==True) (Default value = None)
        figsize: Size of figure (Default value = (8, 2))
        color: Color of tempo curve (Default value = 'k')
        logscale: Use linear (logscale==False) or logartihmic (logscale==True) tempo axis (Default value = False)
        xlabel: Label for x-axis (Default value = 'Time (beats)')
        ylabel: Label for y-axis (Default value = 'Temp (BPM)')
        xlim: Limits for x-axis (Default value = None)
        ylim: Limits for x-axis (Default value = None)
        label: Figure labels when plotting into axis (ax==True) (Default value = '')
        measure_pos: Plot measure positions as spefified (Default value = [])

    Returns:
        fig: figure handle
        ax: axes handle
    """
    fig = None
    if ax is None:
        fig = plt.figure(figsize=figsize)
        ax = plt.subplot(1, 1, 1)
    ax.plot(t_beat, f_tempo, color=color, label=label)
    ax.set_title('Tempo curve')
    if xlim is None:
        xlim = [t_beat[0], t_beat[-1]]
    if ylim is None:
        ylim = [np.min(f_tempo) * 0.9, np.max(f_tempo) * 1.1]
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, which='both')
    if logscale:
        ax.set_yscale('log')
        ax.yaxis.set_major_formatter(ScalarFormatter())
        ax.yaxis.set_minor_formatter(ScalarFormatter())
        # ax.set_yticks([], minor=True)
        # yticks = np.arange(ylim[0], ylim[1]+1, 10)
        # ax.set_yticks(yticks)
    plot_measure(ax, measure_pos)
    return fig, ax


def compute_tempo_curve(score, x, Fs=22050, Fs_beat=10, N=4410, H=2205, shift=0,
                        sigma=np.array([[1, 0], [0, 1], [2, 1], [1, 2], [1, 1]]),
                        win_len_beat=4):
    """Compute a tempo curve

    Notebook: C3/C3S3_MusicAppTempoCurve.ipynb

    Args:
        score (list): Score representation
        x (np.ndarray): Audio signal
        Fs (scalar): Samping rate of audio signal (Default value = 22050)
        Fs_beat (scalar): Sampling rate for beat axis (Default value = 10)
        N (int): Window size for computing audio chromagram (Default value = 4410)
        H (int): Hop size for computing audio chromagram (Default value = 2205)
        shift (int): Cyclic chroma shift applied to audio chromagram (Default value = 0)
        sigma (np.ndarray): Step size set used for DTW
            (Default value = np.array([[1, 0], [0, 1], [2, 1], [1, 2], [1, 1]]))
        win_len_beat (float): Window length (given in beats) used for smoothing tempo curve (Default value = 4)

    Returns:
        f_tempo (np.ndarray): Tempo curve
        t_beat (np.ndarray): Time axis (given in beats)
    """

    # Compute score an audio chromagram
    X_score, t_beat = compute_score_chromagram(score, Fs_beat)
    Fs_X = Fs / H
    X = librosa.feature.chroma_stft(y=x, sr=Fs, norm=2, tuning=0, hop_length=H, n_fft=N)
    X = np.roll(X, shift, axis=0)

    # Apply DTW to compte C, D, P
    C = libfmp.c3.compute_cost_matrix(X, X_score, metric='euclidean')
    D, P = librosa.sequence.dtw(C=C, step_sizes_sigma=sigma)
    P = P[::-1, :]  # reverse P
    P_mod = compute_strict_alignment_path(P)

    # Convert path into beat-time function and interpolte
    t_path_beat = P_mod[:, 1] / Fs_beat
    f_path_sec = P_mod[:, 0] / Fs_X
    f_sec = interp1d(t_path_beat, f_path_sec, kind='linear', fill_value='extrapolate')(t_beat)

    # Compute difference and smooth with Hann window
    f_diff_sec = np.diff(f_sec) * Fs_beat
    pad = np.array([f_diff_sec[-1]])
    f_diff_sec = np.concatenate((f_diff_sec, pad))
    # f_diff_sec = np.concatenate((f_diff_sec, np.array([0]) ))
    filt_len = int(win_len_beat * Fs_beat)
    filt_win = signal.hann(filt_len)
    filt_win = filt_win / np.sum(filt_win)
    f_diff_smooth_sec = scipy.ndimage.filters.convolve(f_diff_sec, filt_win, mode='reflect')

    # Compute tempo curve
    f_tempo = 1. / f_diff_smooth_sec * 60

    return f_tempo, t_beat
