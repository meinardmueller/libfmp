"""
Module: libfmp.c3.c3s1_transposition_tuning
Author: Meinard Müller
License: The MIT license, https://opensource.org/licenses/MIT

This file is part of the FMP Notebooks (https://www.audiolabs-erlangen.de/FMP)
"""
import numpy as np
from matplotlib import pyplot as plt
from scipy.interpolate import interp1d
from scipy import signal
import libfmp.b
import libfmp.c2


def cyclic_shift(C, shift=1):
    """Cyclically shift a chromagram

    Notebook: C3/C3S1_TranspositionTuning.ipynb

    Args:
        C (np.ndarray): Chromagram
        shift (int): Tranposition shift (Default value = 1)

    Returns:
        C_shift (np.ndarray): Cyclically shifted chromagram
    """
    C_shift = np.roll(C, shift=shift, axis=0)
    return C_shift


def compute_freq_distribution(x, Fs, N=16384, gamma=100.0, local=True, filt=True, filt_len=101):
    """Compute an overall frequency distribution

    Notebook: C3/C3S1_TranspositionTuning.ipynb

    Args:
        x (np.ndarray): Signal
        Fs (scalar): Sampling rate
        N (int): Window size (Default value = 16384)
        gamma (float): Constant for logarithmic compression (Default value = 100.0)
        local (bool): Computes STFT and averages; otherwise computes global DFT (Default value = True)
        filt (bool): Applies local frequency averaging and by rectification (Default value = True)
        filt_len (int): Filter length for local frequency averaging (length given in cents) (Default value = 101)

    Returns:
        v (np.ndarray): Vector representing an overall frequency distribution
        F_coef_cents (np.ndarray): Frequency axis (given in cents)
    """
    if local:
        # Compute an STFT and sum over time
        if N > len(x)//2:
            raise Exception('The signal length (%d) should be twice as long as the window length (%d)' % (len(x), N))
        Y, T_coef, F_coef = libfmp.c2.stft_convention_fmp(x=x, Fs=Fs, N=N, H=N//2, mag=True, gamma=gamma)
        # Error "range() arg 3 must not be zero" occurs when N is too large. Why?
        Y = np.sum(Y, axis=1)
    else:
        # Compute a single DFT for the entire signal
        N = len(x)
        Y = np.abs(np.fft.fft(x)) / Fs
        Y = Y[:N//2+1]
        Y = np.log(1 + gamma * Y)
        # Y = libfmp.c3.log_compression(Y, gamma=100)
        F_coef = np.arange(N // 2 + 1).astype(float) * Fs / N

    # Convert linearly spaced frequency axis in logarithmic axis (given in cents)
    # The minimum frequency F_min corresponds 0 cents.
    f_pitch = lambda p: 440 * 2 ** ((p - 69) / 12)
    p_min = 24               # C1, MIDI pitch 24
    F_min = f_pitch(p_min)   # 32.70 Hz
    p_max = 108              # C8, MIDI pitch 108
    F_max = f_pitch(p_max)   # 4186.01 Hz
    F_coef_log, F_coef_cents = libfmp.c2.compute_f_coef_log(R=1, F_min=F_min, F_max=F_max)
    Y_int = interp1d(F_coef, Y, kind='cubic', fill_value='extrapolate')(F_coef_log)
    v = Y_int / np.max(Y_int)

    if filt:
        # Subtract local average and rectify
        filt_kernel = np.ones(filt_len)
        Y_smooth = signal.convolve(Y_int, filt_kernel, mode='same') / filt_len
        # Y_smooth = signal.medfilt(Y_int, filt_len)
        Y_rectified = Y_int - Y_smooth
        Y_rectified[Y_rectified < 0] = 0
        v = Y_rectified / np.max(Y_rectified)

    return v, F_coef_cents


def template_comb(M, theta=0):
    """Compute a comb template on a pitch axis

    Notebook: C3/C3S1_TranspositionTuning.ipynb

    Args:
        M (int): Length template (given in cents)
        theta (int): Shift parameter (given in cents); -50 <= theta < 50 (Default value = 0)

    Returns:
        template (np.ndarray): Comb template shifted by theta
    """
    template = np.zeros(M)
    peak_positions = (np.arange(0, M, 100) + theta)
    peak_positions = np.intersect1d(peak_positions, np.arange(M)).astype(int)
    template[peak_positions] = 1
    return template


def tuning_similarity(v):
    """Compute tuning similarity

    Notebook: C3/C3S1_TranspositionTuning.ipynb

    Args:
        v (np.ndarray): Vector representing an overall frequency distribution

    Returns:
        theta_axis (np.ndarray): Axis consisting of all tuning parameters -50 <= theta < 50
        sim (np.ndarray): Similarity values for all tuning parameters
        ind_max (int): Maximizing index
        theta_max (int): Maximizing tuning parameter
        template_max (np.ndarray): Similiarty-maximizing comb template
    """
    theta_axis = np.arange(-50, 50)  # Axis (given in cents)
    num_theta = len(theta_axis)
    sim = np.zeros(num_theta)
    M = len(v)
    for i in range(num_theta):
        theta = theta_axis[i]
        template = template_comb(M=M, theta=theta)
        sim[i] = np.inner(template, v)
    sim = sim / np.max(sim)
    ind_max = np.argmax(sim)
    theta_max = theta_axis[ind_max]
    template_max = template_comb(M=M, theta=theta_max)
    return theta_axis, sim, ind_max, theta_max, template_max


def plot_tuning_similarity(sim, theta_axis, theta_max, ax=None, title=None, figsize=(4, 3)):
    """Plots tuning similarity

    Notebook: C3/C3S1_TranspositionTuning.ipynb

    Args:
        sim: Similarity values
        theta_axis: Axis consisting of cent values [-50:49]
        theta_max: Maximizing tuning parameter
        ax: Axis (in case of ax=None, figure is generated) (Default value = None)
        title: Title of figure (or subplot) (Default value = None)
        figsize: Size of figure (only used when ax=None) (Default value = (4, 3))

    Returns:
        fig: Handle for figure
        ax: Handle for axes
        line: handle for line plot
    """
    fig = None
    if ax is None:
        fig = plt.figure(figsize=figsize)
        ax = plt.subplot(1, 1, 1)
    if title is None:
        title = 'Estimated tuning: %d cents' % theta_max
    line = ax.plot(theta_axis, sim, 'k')
    ax.set_xlim([theta_axis[0], theta_axis[-1]])
    ax.set_ylim([0, 1.1])
    ax.plot([theta_max, theta_max], [0, 1.1], 'r')
    ax.set_xlabel('Tuning parameter (cents)')
    ax.set_ylabel('Similarity')
    ax.set_title(title)
    if fig is not None:
        plt.tight_layout()
    return fig, ax, line


def plot_freq_vector_template(v, F_coef_cents, template_max, theta_max, ax=None, title=None, figsize=(8, 3)):
    """Plots frequency distribution and similarity-maximizing template

    Notebook: C3/C3S1_TranspositionTuning.ipynb

    Args:
        v: Vector representing an overall frequency distribution
        F_coef_cents: Frequency axis
        template_max: Similarity-maximizing template
        theta_max: Maximizing tuning parameter
        ax: Axis (in case of ax=None, figure is generated) (Default value = None)
        title: Title of figure (or subplot) (Default value = None)
        figsize: Size of figure (only used when ax=None) (Default value = (8, 3))

    Returns:
        fig: Handle for figure
        ax: Handle for axes
        line: handle for line plot
    """
    fig = None
    if ax is None:
        fig = plt.figure(figsize=figsize)
        ax = plt.subplot(1, 1, 1)
    if title is None:
        title = r'Frequency distribution with maximizing comb template ($\theta$ = %d cents)' % theta_max
    line = ax.plot(F_coef_cents, v, c='k', linewidth=1)
    ax.set_xlim([F_coef_cents[0], F_coef_cents[-1]])
    ax.set_ylim([0, 1.1])
    x_ticks_freq = np.array([0, 1200, 2400, 3600, 4800, 6000, 7200, 8000])
    ax.plot(F_coef_cents, template_max * 1.1, 'r:', linewidth=0.5)
    ax.set_xticks(x_ticks_freq)
    ax.set_xlabel('Frequency (cents)')
    plt.title(title)
    if fig is not None:
        plt.tight_layout()
    return fig, ax, line
