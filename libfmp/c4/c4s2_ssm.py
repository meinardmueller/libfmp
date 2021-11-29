"""
Module: libfmp.c4.c4s2_ssm
Author: Meinard Müller, David Kopyto
License: The MIT license, https://opensource.org/licenses/MIT

This file is part of the FMP Notebooks (https://www.audiolabs-erlangen.de/FMP)
"""
import numpy as np
import librosa
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap
from numba import jit

import libfmp.b
import libfmp.c3
import libfmp.c4


@jit(nopython=True)
def compute_sm_dot(X, Y):
    """Computes similarty matrix from feature sequences using dot (inner) product

    Notebook: C4/C4S2_SSM.ipynb

    Args:
        X (np.ndarray): First sequence
        Y (np.ndarray): Second Sequence

    Returns:
        S (float): Dot product
    """
    S = np.dot(np.transpose(X), Y)
    return S


def plot_feature_ssm(X, Fs_X, S, Fs_S, ann, duration, color_ann=None,
                     title='', label='Time (seconds)', time=True,
                     figsize=(5, 6), fontsize=10, clim_X=None, clim=None):
    """Plot SSM along with feature representation and annotations (standard setting is time in seconds)

    Notebook: C4/C4S2_SSM.ipynb

    Args:
        X: Feature representation
        Fs_X: Feature rate of ``X``
        S: Similarity matrix (SM)
        Fs_S: Feature rate of ``S``
        ann: Annotaions
        duration: Duration
        color_ann: Color annotations (see :func:`libfmp.b.b_plot.plot_segments`) (Default value = None)
        title: Figure title (Default value = '')
        label: Label for time axes (Default value = 'Time (seconds)')
        time: Display time axis ticks or not (Default value = True)
        figsize: Figure size (Default value = (5, 6))
        fontsize: Font size (Default value = 10)
        clim_X: Color limits for matrix X (Default value = None)
        clim: Color limits for matrix ``S`` (Default value = None)

    Returns:
        fig: Handle for figure
        ax: Handle for axes
    """
    cmap = libfmp.b.compressed_gray_cmap(alpha=-10)
    fig, ax = plt.subplots(3, 3, gridspec_kw={'width_ratios': [0.1, 1, 0.05],
                                              'wspace': 0.2,
                                              'height_ratios': [0.3, 1, 0.1]},
                           figsize=figsize)
    libfmp.b.plot_matrix(X, Fs=Fs_X, ax=[ax[0, 1], ax[0, 2]], clim=clim_X,
                         xlabel='', ylabel='', title=title)
    ax[0, 0].axis('off')
    libfmp.b.plot_matrix(S, Fs=Fs_S, ax=[ax[1, 1], ax[1, 2]], cmap=cmap, clim=clim,
                         title='', xlabel='', ylabel='', colorbar=True)
    ax[1, 1].set_xticks([])
    ax[1, 1].set_yticks([])
    libfmp.b.plot_segments(ann, ax=ax[2, 1], time_axis=time, fontsize=fontsize,
                           colors=color_ann,
                           time_label=label, time_max=duration*Fs_X)
    ax[2, 2].axis('off')
    ax[2, 0].axis('off')
    libfmp.b.plot_segments(ann, ax=ax[1, 0], time_axis=time, fontsize=fontsize,
                           direction='vertical', colors=color_ann,
                           time_label=label, time_max=duration*Fs_X)
    return fig, ax


@jit(nopython=True)
def filter_diag_sm(S, L):
    """Path smoothing of similarity matrix by forward filtering along main diagonal

    Notebook: C4/C4S2_SSM-PathEnhancement.ipynb

    Args:
        S (np.ndarray): Similarity matrix (SM)
        L (int): Length of filter

    Returns:
        S_L (np.ndarray): Smoothed SM
    """
    N = S.shape[0]
    M = S.shape[1]
    S_L = np.zeros((N, M))
    S_extend_L = np.zeros((N + L, M + L))
    S_extend_L[0:N, 0:M] = S
    for pos in range(0, L):
        S_L = S_L + S_extend_L[pos:(N + pos), pos:(M + pos)]
    S_L = S_L / L
    return S_L


def subplot_matrix_colorbar(S, fig, ax, title='', Fs=1,
                            xlabel='Time (seconds)', ylabel='Time (seconds)',
                            clim=None, xlim=None, ylim=None, cmap=None, interpolation='nearest'):
    """Visualization function for showing zoomed sections of matrices

    Notebook: C4/C4S2_SSM-PathEnhancement.ipynb

    Args:
        S: Similarity matrix (SM)
        fig: Figure handle
        ax: Axes handle
        title: Title for figure (Default value = '')
        Fs: Feature rate (Default value = 1)
        xlabel: Label for x-axis (Default value = 'Time (seconds)')
        ylabel: Label for y-axis (Default value = 'Time (seconds)')
        clim: Color limits (Default value = None)
        xlim: Limits for x-axis (Default value = None)
        ylim: Limits for x-axis (Default value = None)
        cmap: Colormap for imshow (Default value = None)
        interpolation: Interpolation value for imshow (Default value = 'nearest')

    Returns:
        im: Imshow handle
    """
    if cmap is None:
        cmap = libfmp.b.compressed_gray_cmap(alpha=-100)
    len_sec = S.shape[0] / Fs
    extent = [0, len_sec, 0, len_sec]
    im = ax.imshow(S, aspect='auto', extent=extent, cmap=cmap,  origin='lower', interpolation=interpolation)
    fig.sca(ax)
    fig.colorbar(im)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)
    if clim is not None:
        im.set_clim(clim)
    return im


@jit(nopython=True)
def compute_tempo_rel_set(tempo_rel_min, tempo_rel_max, num):
    """Compute logarithmically spaced relative tempo values

    Notebook: C4/C4S2_SSM-PathEnhancement.ipynb

    Args:
        tempo_rel_min (float): Minimum relative tempo
        tempo_rel_max (float): Maximum relative tempo
        num (int): Number of relative tempo values (inlcuding the min and max)

    Returns:
        tempo_rel_set (np.ndarray): Set of relative tempo values
    """
    tempo_rel_set = np.exp(np.linspace(np.log(tempo_rel_min), np.log(tempo_rel_max), num))
    return tempo_rel_set


@jit(nopython=True)
def filter_diag_mult_sm(S, L=1, tempo_rel_set=np.asarray([1]), direction=0):
    """Path smoothing of similarity matrix by filtering in forward or backward direction
    along various directions around main diagonal.
    Note: Directions are simulated by resampling one axis using relative tempo values

    Notebook: C4/C4S2_SSM-PathEnhancement.ipynb

    Args:
        S (np.ndarray): Self-similarity matrix (SSM)
        L (int): Length of filter (Default value = 1)
        tempo_rel_set (np.ndarray): Set of relative tempo values (Default value = np.asarray([1]))
        direction (int): Direction of smoothing (0: forward; 1: backward) (Default value = 0)

    Returns:
        S_L_final (np.ndarray): Smoothed SM
    """
    N = S.shape[0]
    M = S.shape[1]
    num = len(tempo_rel_set)
    S_L_final = np.zeros((N, M))

    for s in range(0, num):
        M_ceil = int(np.ceil(M / tempo_rel_set[s]))
        resample = np.multiply(np.divide(np.arange(1, M_ceil+1), M_ceil), M)
        np.around(resample, 0, resample)
        resample = resample - 1
        index_resample = np.maximum(resample, np.zeros(len(resample))).astype(np.int64)
        S_resample = S[:, index_resample]

        S_L = np.zeros((N, M_ceil))
        S_extend_L = np.zeros((N + L, M_ceil + L))

        # Forward direction
        if direction == 0:
            S_extend_L[0:N, 0:M_ceil] = S_resample
            for pos in range(0, L):
                S_L = S_L + S_extend_L[pos:(N + pos), pos:(M_ceil + pos)]

        # Backward direction
        if direction == 1:
            S_extend_L[L:(N+L), L:(M_ceil+L)] = S_resample
            for pos in range(0, L):
                S_L = S_L + S_extend_L[(L-pos):(N + L - pos), (L-pos):(M_ceil + L - pos)]

        S_L = S_L / L
        resample = np.multiply(np.divide(np.arange(1, M+1), M), M_ceil)
        np.around(resample, 0, resample)
        resample = resample - 1
        index_resample = np.maximum(resample, np.zeros(len(resample))).astype(np.int64)

        S_resample_inv = S_L[:, index_resample]
        S_L_final = np.maximum(S_L_final, S_resample_inv)

    return S_L_final


@jit(nopython=True)
def shift_cyc_matrix(X, shift=0):
    """Cyclic shift of features matrix along first dimension

    Notebook: C4/C4S2_SSM-TranspositionInvariance.ipynb

    Args:
        X (np.ndarray): Feature respresentation
        shift (int): Number of bins to be shifted (Default value = 0)

    Returns:
        X_cyc (np.ndarray): Cyclically shifted feature matrix
    """
    # Note: X_cyc = np.roll(X, shift=shift, axis=0) does to work for jit
    K, N = X.shape
    shift = np.mod(shift, K)
    X_cyc = np.zeros((K, N))
    X_cyc[shift:K, :] = X[0:K-shift, :]
    X_cyc[0:shift, :] = X[K-shift:K, :]
    return X_cyc


# @jit(nopython=True)
def compute_sm_ti(X, Y, L=1, tempo_rel_set=np.asarray([1]), shift_set=np.asarray([0]), direction=2):
    """Compute enhanced similaity matrix by applying path smoothing and transpositions

    Notebook: C4/C4S2_SSM-TranspositionInvariance.ipynb

    Args:
        X (np.ndarray): First feature sequence
        Y (np.ndarray): Second feature sequence
        L (int): Length of filter (Default value = 1)
        tempo_rel_set (np.ndarray): Set of relative tempo values (Default value = np.asarray([1]))
        shift_set (np.ndarray): Set of shift indices (Default value = np.asarray([0]))
        direction (int): Direction of smoothing (0: forward; 1: backward; 2: both directions) (Default value = 2)

    Returns:
        S_TI (np.ndarray): Transposition-invariant SM
        I_TI (np.ndarray): Transposition index matrix
    """
    for shift in shift_set:
        Y_cyc = shift_cyc_matrix(Y, shift)
        S_cyc = libfmp.c4.compute_sm_dot(X, Y_cyc)

        if direction == 0:
            S_cyc = libfmp.c4.filter_diag_mult_sm(S_cyc, L, tempo_rel_set, direction=0)
        if direction == 1:
            S_cyc = libfmp.c4.filter_diag_mult_sm(S_cyc, L, tempo_rel_set, direction=1)
        if direction == 2:
            S_forward = libfmp.c4.filter_diag_mult_sm(S_cyc, L, tempo_rel_set=tempo_rel_set, direction=0)
            S_backward = libfmp.c4.filter_diag_mult_sm(S_cyc, L, tempo_rel_set=tempo_rel_set, direction=1)
            S_cyc = np.maximum(S_forward, S_backward)
        if shift == shift_set[0]:
            S_TI = S_cyc
            I_TI = np.ones((S_cyc.shape[0], S_cyc.shape[1])) * shift
        else:
            # jit does not like the following lines
            # I_greater = np.greater(S_cyc, S_TI)
            # I_greater = (S_cyc > S_TI)
            I_TI[S_cyc > S_TI] = shift
            S_TI = np.maximum(S_cyc, S_TI)

    return S_TI, I_TI


def subplot_matrix_ti_colorbar(S, fig, ax, title='', Fs=1, xlabel='Time (seconds)', ylabel='Time (seconds)',
                               clim=None, xlim=None, ylim=None, cmap=None, alpha=1, interpolation='nearest',
                               ind_zero=False):
    """Visualization function for showing transposition index matrix

    Notebook: C4/C4S2_SSM-TranspositionInvariance.ipynb

    Args:
        S: Self-similarity matrix (SSM)
        fig: Figure handle
        ax: Axes handle
        title: Title for figure (Default value = '')
        Fs: Feature rate (Default value = 1)
        xlabel: Label for x-axis (Default value = 'Time (seconds)')
        ylabel: Label for y-axis (Default value = 'Time (seconds)')
        clim: Color limits (Default value = None)
        xlim: Limits for x-axis (Default value = None)
        ylim: Limits for y-axis (Default value = None)
        cmap: Color map (Default value = None)
        alpha: Alpha value for imshow (Default value = 1)
        interpolation: Interpolation value for imshow (Default value = 'nearest')
        ind_zero: Use white (True) or black (False) color for index zero (Default value = False)

    Returns:
        im: Imshow handle
    """
    if cmap is None:
        color_ind_zero = np.array([0, 0, 0, 1])
        if ind_zero == 0:
            color_ind_zero = np.array([0, 0, 0, 1])
        else:
            color_ind_zero = np.array([1, 1, 1, 1])
        colorList = np.array([color_ind_zero, [1, 1, 0, 1],  [0, 0.7, 0, 1],  [1, 0, 1, 1],  [0, 0, 1, 1],
                             [1, 0, 0, 1], [0, 0, 0, 0.5], [1, 0, 0, 0.3], [0, 0, 1, 0.3], [1, 0, 1, 0.3],
                             [0, 0.7, 0, 0.3], [1, 1, 0, 0.3]])
        cmap = ListedColormap(colorList)
    len_sec = S.shape[0] / Fs
    extent = [0, len_sec, 0, len_sec]
    im = ax.imshow(S, aspect='auto', extent=extent, cmap=cmap,  origin='lower', alpha=alpha,
                   interpolation=interpolation)
    if clim is None:
        im.set_clim(vmin=-0.5, vmax=11.5)
    fig.sca(ax)
    ax_cb = fig.colorbar(im)
    ax_cb.set_ticks(np.arange(0, 12, 1))
    ax_cb.set_ticklabels(np.arange(0, 12, 1))
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)
    return im


def compute_sm_from_filename(fn_wav, L=21, H=5, L_smooth=16, tempo_rel_set=np.array([1]),
                             shift_set=np.array([0]), strategy='relative', scale=True, thresh=0.15,
                             penalty=0.0, binarize=False):
    """Compute an SSM

    Notebook: C4/C4S2_SSM-Thresholding.ipynb

    Args:
        fn_wav (str): Path and filename of wav file
        L (int): Length of smoothing filter (Default value = 21)
        H (int): Downsampling factor (Default value = 5)
        L_smooth (int): Length of filter (Default value = 16)
        tempo_rel_set (np.ndarray):  Set of relative tempo values (Default value = np.array([1]))
        shift_set (np.ndarray): Set of shift indices (Default value = np.array([0]))
        strategy (str): Thresholding strategy (see :func:`libfmp.c4.c4s2_ssm.compute_sm_ti`)
            (Default value = 'relative')
        scale (bool): If scale=True, then scaling of positive values to range [0,1] (Default value = True)
        thresh (float): Treshold (meaning depends on strategy) (Default value = 0.15)
        penalty (float): Set values below treshold to value specified (Default value = 0.0)
        binarize (bool): Binarizes final matrix (positive: 1; otherwise: 0) (Default value = False)

    Returns:
        x (np.ndarray): Audio signal
        x_duration (float): Duration of audio signal (seconds)
        X (np.ndarray): Feature sequence
        Fs_feature (scalar): Feature rate
        S_thresh (np.ndarray): SSM
        I (np.ndarray): Index matrix
    """
    # Waveform
    Fs = 22050
    x, Fs = librosa.load(fn_wav, Fs)
    x_duration = x.shape[0] / Fs

    # Chroma Feature Sequence and SSM (10 Hz)
    C = librosa.feature.chroma_stft(y=x, sr=Fs, tuning=0, norm=2, hop_length=2205, n_fft=4410)
    Fs_C = Fs / 2205

    # Chroma Feature Sequence and SSM
    X, Fs_feature = libfmp.c3.smooth_downsample_feature_sequence(C, Fs_C, filt_len=L, down_sampling=H)
    X = libfmp.c3.normalize_feature_sequence(X, norm='2', threshold=0.001)

    # Compute SSM
    S, I = libfmp.c4.compute_sm_ti(X, X, L=L_smooth, tempo_rel_set=tempo_rel_set, shift_set=shift_set, direction=2)
    S_thresh = libfmp.c4.threshold_matrix(S, thresh=thresh, strategy=strategy,
                                          scale=scale, penalty=penalty, binarize=binarize)
    return x, x_duration, X, Fs_feature, S_thresh, I
