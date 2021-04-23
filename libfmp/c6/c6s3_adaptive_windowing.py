"""
Module: libfmp.c6.c6s3_adaptive_windowing
Author: Meinard Müller
License: The MIT license, https://opensource.org/licenses/MIT

This file is part of the FMP Notebooks (https://www.audiolabs-erlangen.de/FMP)
"""

import numpy as np
from matplotlib import pyplot as plt
import libfmp.b


def plot_beat_grid(B_sec, ax, color='r', linestyle=':', linewidth=1):
    """Plot beat grid (given in seconds) into axis

    Notebook: C6/C6S3_AdaptiveWindowing.ipynb

    Args:
        B_sec: Beat grid
        ax: Axes for plotting
        color: Color of lines (Default value = 'r')
        linestyle: Style of lines (Default value = ':')
        linewidth: Width of lines (Default value = 1)
    """
    for b in B_sec:
        ax.axvline(x=b, color=color, linestyle=linestyle, linewidth=linewidth)


def adaptive_windowing(X, B, neigborhood=1, add_start=False, add_end=False):
    """Apply adaptive windowing [FMP, Section 6.3.3]

    Notebook: C6/C6S3_AdaptiveWindowing.ipynb

    Args:
        X (np.ndarray): Feature sequence
        B (np.ndarray): Beat sequence (spefied in frames)
        neigborhood (float): Parameter specifying relative range considered for windowing (Default value = 1)
        add_start (bool): Add first index of X to beat sequence (if not existent) (Default value = False)
        add_end (bool): Add last index of X to beat sequence (if not existent) (Default value = False)

    Returns:
        X_adapt (np.ndarray): Feature sequence adapted to beat sequence
        B_s (np.ndarray): Sequence specifying start (in frames) of window sections
        B_t (np.ndarray): Sequence specifying end (in frames) of window sections
    """
    len_X = X.shape[1]
    max_B = np.max(B)
    if max_B > len_X:
        print('Beat exceeds length of features sequence (b=%d, |X|=%d)' % (max_B, len_X))
        B = B[B < len_X]
    if add_start:
        if B[0] > 0:
            B = np.insert(B, 0, 0)
    if add_end:
        if B[-1] < len_X:
            B = np.append(B, len_X)
    X_adapt = np.zeros((X.shape[0], len(B)-1))
    B_s = np.zeros(len(B)-1).astype(int)
    B_t = np.zeros(len(B)-1).astype(int)
    for b in range(len(B)-1):
        s = B[b]
        t = B[b+1]
        reduce = np.floor((1 - neigborhood)*(t-s+1)/2).astype(int)
        s = s + reduce
        t = t - reduce
        if s == t:
            t = t + 1
        X_slice = X[:, range(s, t)]
        X_adapt[:, b] = np.mean(X_slice, axis=1)
        B_s[b] = s
        B_t[b] = t
    return X_adapt, B_s, B_t


def compute_plot_adaptive_windowing(x, Fs, H, X, B, neigborhood=1, add_start=False, add_end=False):
    """Compute and plot process for adaptive windowing [FMP, Section 6.3.3]

    Notebook: C6/C6S3_AdaptiveWindowing.ipynb

    Args:
        x (np.ndarray): Signal
        Fs (scalar): Sample Rate
        H (int): Hop size
        X (int): Feature sequence
        B (np.ndarray): Beat sequence (spefied in frames)
        neigborhood (float): Parameter specifying relative range considered for windowing (Default value = 1)
        add_start (bool): Add first index of X to beat sequence (if not existent) (Default value = False)
        add_end (bool): Add last index of X to beat sequence (if not existent) (Default value = False)

    Returns:
        X_adapt (np.ndarray): Feature sequence adapted to beat sequence
    """
    X_adapt, B_s, B_t = adaptive_windowing(X, B, neigborhood=neigborhood,
                                           add_start=add_start, add_end=add_end)

    fig, ax = plt.subplots(2, 2, gridspec_kw={'width_ratios': [1, 0.03],
                                              'height_ratios': [1, 3]}, figsize=(10, 4))

    libfmp.b.plot_signal(x, Fs, ax=ax[0, 0], title=r'Adaptive windowing using $\lambda = %0.2f$' % neigborhood)
    ax[0, 1].set_axis_off()
    plot_beat_grid(B_s * H / Fs, ax[0, 0], color='b')
    plot_beat_grid(B_t * H / Fs, ax[0, 0], color='g')
    plot_beat_grid(B * H / Fs, ax[0, 0], color='r')
    for k in range(len(B_s)):
        ax[0, 0].fill_between([B_s[k] * H / Fs, B_t[k] * H / Fs], -1, 1, facecolor='red', alpha=0.1)

    libfmp.b.plot_matrix(X_adapt, ax=[ax[1, 0], ax[1, 1]], xlabel='Time (frames)', ylabel='Frequency (bins)')
    plt.tight_layout()
    return X_adapt
