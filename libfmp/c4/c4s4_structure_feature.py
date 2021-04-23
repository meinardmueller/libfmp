"""
Module: libfmp.c4.c4s4_structure_feature
Author: Meinard Müller, Tim Zunner
License: The MIT license, https://opensource.org/licenses/MIT

This file is part of the FMP Notebooks (https://www.audiolabs-erlangen.de/FMP)
"""
import numpy as np
from matplotlib import pyplot as plt

import libfmp.b


def compute_time_lag_representation(S, circular=True):
    """Computation of (circular) time-lag representation

    Notebook: C4/C4S4_StructureFeature.ipynb

    Args:
        S (np.ndarray): Self-similarity matrix
        circular (bool): Computes circular version (Default value = True)

    Returns:
        L (np.ndarray): (Circular) time-lag representation of S
    """
    N = S.shape[0]
    if circular:
        L = np.zeros((N, N))
        for n in range(N):
            L[:, n] = np.roll(S[:, n], -n)
    else:
        L = np.zeros((2*N-1, N))
        for n in range(N):
            L[((N-1)-n):((2*N)-1-n), n] = S[:, n]
    return L


def novelty_structure_feature(L, padding=True):
    """Computation of the novelty function from a circular time-lag representation

    Notebook: C4/C4S4_StructureFeature.ipynb

    Args:
        L (np.ndarray): Circular time-lag representation
        padding (bool): Padding the result with the value zero (Default value = True)

    Returns:
        nov (np.ndarray): Novelty function
    """
    N = L.shape[0]
    if padding:
        nov = np.zeros(N)
    else:
        nov = np.zeros(N-1)
    for n in range(N-1):
        nov[n] = np.linalg.norm(L[:, n+1] - L[:, n])
    return nov


def plot_ssm_structure_feature_nov(S, L, nov, Fs=1, figsize=(10, 3), ann=[], color_ann=[]):
    """Plotting an SSM, structure features, and a novelty function

    Notebook: C4/C4S4_StructureFeature.ipynb

    Args:
        S: SSM
        L: Circular time-lag representation
        nov: Novelty function
        Fs: Feature rate (indicated in title of SSM) (Default value = 1)
        figsize: Figure size (Default value = (10, 3))
        ann: Annotations (Default value = [])
        color_ann: Colors used for annotations (see :func:`libfmp.b.b_plot.plot_segments`) (Default value = [])

    Returns:
        ax1: First subplot
        ax2: Second subplot
        ax3: Third subplot
    """
    plt.figure(figsize=figsize)
    ax1 = plt.subplot(131)
    if Fs == 1:
        title = 'SSM'
    else:
        title = 'SSM (Fs = %d)' % Fs
    fig_im, ax_im, im = libfmp.b.plot_matrix(S, ax=[ax1], title=title,
                                             xlabel='Time (frames)', ylabel='Time (frames)')
    if ann:
        libfmp.b.plot_segments_overlay(ann, ax=ax_im[0], edgecolor='k',
                                       print_labels=False, colors=color_ann, alpha=0.05)

    ax2 = plt.subplot(132)
    fig_im, ax_im, im = libfmp.b.plot_matrix(L, ax=[ax2], title='Structure features',
                                             xlabel='Time (frames)', ylabel='Lag (frames)', colorbar=True)
    if ann:
        libfmp.b.plot_segments_overlay(ann, ax=ax_im[0], edgecolor='k', ylim=False,
                                       print_labels=False, colors=color_ann, alpha=0.05)

    ax3 = plt.subplot(133)
    fig, ax, im = libfmp.b.plot_signal(nov, ax=ax3, title='Novelty function',
                                       xlabel='Time (frames)', color='k')
    if ann:
        libfmp.b.plot_segments_overlay(ann, ax=ax, edgecolor='k', colors=color_ann, alpha=0.05)
    plt.tight_layout()
    return ax1, ax2, ax3
