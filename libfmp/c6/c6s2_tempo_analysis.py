"""
Module: libfmp.c6.c6s2_tempo_analysis
Author: Meinard Müller, Angel Villar-Corrales
License: The MIT license, https://opensource.org/licenses/MIT

This file is part of the FMP Notebooks (https://www.audiolabs-erlangen.de/FMP)
"""
import numpy as np
import librosa
from scipy import signal
from scipy.interpolate import interp1d
from matplotlib import pyplot as plt
from numba import jit
import IPython.display as ipd

import libfmp.b
import libfmp.c6


@jit(nopython=True)
def compute_tempogram_fourier(x, Fs, N, H, Theta=np.arange(30, 601, 1)):
    """Compute Fourier-based tempogram [FMP, Section 6.2.2]

    Notebook: C6/C6S2_TempogramFourier.ipynb

    Args:
        x (np.ndarray): Input signal
        Fs (scalar): Sampling rate
        N (int): Window length
        H (int): Hop size
        Theta (np.ndarray): Set of tempi (given in BPM) (Default value = np.arange(30, 601, 1))

    Returns:
        X (np.ndarray): Tempogram
        T_coef (np.ndarray): Time axis (seconds)
        F_coef_BPM (np.ndarray): Tempo axis (BPM)
    """
    win = np.hanning(N)
    N_left = N // 2
    L = x.shape[0]
    L_left = N_left
    L_right = N_left
    L_pad = L + L_left + L_right
    # x_pad = np.pad(x, (L_left, L_right), 'constant')  # doesn't work with jit
    x_pad = np.concatenate((np.zeros(L_left), x, np.zeros(L_right)))
    t_pad = np.arange(L_pad)
    M = int(np.floor(L_pad - N) / H) + 1
    K = len(Theta)
    X = np.zeros((K, M), dtype=np.complex_)

    for k in range(K):
        omega = (Theta[k] / 60) / Fs
        exponential = np.exp(-2 * np.pi * 1j * omega * t_pad)
        x_exp = x_pad * exponential
        for n in range(M):
            t_0 = n * H
            t_1 = t_0 + N
            X[k, n] = np.sum(win * x_exp[t_0:t_1])
        T_coef = np.arange(M) * H / Fs
        F_coef_BPM = Theta
    return X, T_coef, F_coef_BPM


def compute_sinusoid_optimal(c, tempo, n, Fs, N, H):
    """Compute windowed sinusoid with optimal phase

    Notebook: C6/C6S2_TempogramFourier.ipynb

    Args:
        c (complex): Coefficient of tempogram (c=X(k,n))
        tempo (float): Tempo parameter corresponding to c (tempo=F_coef_BPM[k])
        n (int): Frame parameter of c
        Fs (scalar): Sampling rate
        N (int): Window length
        H (int): Hop size

    Returns:
        kernel (np.ndarray): Windowed sinusoid
        t_kernel (np.ndarray): Time axis (samples) of kernel
        t_kernel_sec (np.ndarray): Time axis (seconds) of kernel
    """
    win = np.hanning(N)
    N_left = N // 2
    omega = (tempo / 60) / Fs
    t_0 = n * H
    t_1 = t_0 + N
    phase = - np.angle(c) / (2 * np.pi)
    t_kernel = np.arange(t_0, t_1)
    kernel = win * np.cos(2 * np.pi * (t_kernel*omega - phase))
    t_kernel_sec = (t_kernel - N_left) / Fs
    return kernel, t_kernel, t_kernel_sec


def plot_signal_kernel(x, t_x, kernel, t_kernel, xlim=None, figsize=(8, 2), title=None):
    """Visualize signal and local kernel

    Notebook: C6/C6S2_TempogramFourier.ipynb

    Args:
        x: Signal
        t_x: Time axis of x (given in seconds)
        kernel: Local kernel
        t_kernel: Time axis of kernel (given in seconds)
        xlim: Limits for x-axis (Default value = None)
        figsize: Figure size (Default value = (8, 2))
        title: Title of figure (Default value = None)

    Returns:
        fig: Matplotlib figure handle
    """
    if xlim is None:
        xlim = [t_x[0], t_x[-1]]
    fig = plt.figure(figsize=figsize)
    plt.plot(t_x, x, 'k')
    plt.plot(t_kernel, kernel, 'r')
    plt.title(title)
    plt.xlim(xlim)
    plt.tight_layout()
    return fig


# @jit(nopython=True)  # not possible because of np.correlate with mode='full'
def compute_autocorrelation_local(x, Fs, N, H, norm_sum=True):
    """Compute local autocorrelation [FMP, Section 6.2.3]

    Notebook: C6/C6S2_TempogramAutocorrelation.ipynb

    Args:
        x (np.ndarray): Input signal
        Fs (scalar): Sampling rate
        N (int): Window length
        H (int): Hop size
        norm_sum (bool): Normalizes by the number of summands in local autocorrelation (Default value = True)

    Returns:
        A (np.ndarray): Time-lag representation
        T_coef (np.ndarray): Time axis (seconds)
        F_coef_lag (np.ndarray): Lag axis
    """
    # L = len(x)
    L_left = round(N / 2)
    L_right = L_left
    x_pad = np.concatenate((np.zeros(L_left), x, np.zeros(L_right)))
    L_pad = len(x_pad)
    M = int(np.floor(L_pad - N) / H) + 1
    A = np.zeros((N, M))
    win = np.ones(N)
    if norm_sum is True:
        lag_summand_num = np.arange(N, 0, -1)
    for n in range(M):
        t_0 = n * H
        t_1 = t_0 + N
        x_local = win * x_pad[t_0:t_1]
        r_xx = np.correlate(x_local, x_local, mode='full')
        r_xx = r_xx[N-1:]
        if norm_sum is True:
            r_xx = r_xx / lag_summand_num
        A[:, n] = r_xx
    Fs_A = Fs / H
    T_coef = np.arange(A.shape[1]) / Fs_A
    F_coef_lag = np.arange(N) / Fs
    return A, T_coef, F_coef_lag


def plot_signal_local_lag(x, t_x, local_lag, t_local_lag, lag, xlim=None, figsize=(8, 1.5), title=''):
    """Visualize signal and local lag [FMP, Figure 6.14]

    Notebook: C6/C6S2_TempogramAutocorrelation.ipynb

    Args:
        x: Signal
        t_x: Time axis of x (given in seconds)
        local_lag: Local lag
        t_local_lag: Time axis of kernel (given in seconds)
        lag: Lag (given in seconds)
        xlim: Limits for x-axis (Default value = None)
        figsize: Figure size (Default value = (8, 1.5))
        title: Title of figure (Default value = '')

    Returns:
        fig: Matplotlib figure handle
    """
    if xlim is None:
        xlim = [t_x[0], t_x[-1]]
    fig = plt.figure(figsize=figsize)
    plt.plot(t_x, x, 'k:', linewidth=0.5)
    plt.plot(t_local_lag, local_lag, 'k', linewidth=3.0)
    plt.plot(t_local_lag+lag, local_lag, 'r', linewidth=2)
    plt.title(title)
    plt.ylim([0, 1.1 * np.max(x)])
    plt.xlim(xlim)
    plt.tight_layout()
    return fig


# @jit(nopython=True)
def compute_tempogram_autocorr(x, Fs, N, H, norm_sum=False, Theta=np.arange(30, 601)):
    """Compute autocorrelation-based tempogram

    Notebook: C6/C6S2_TempogramAutocorrelation.ipynb

    Args:
        x (np.ndarray): Input signal
        Fs (scalar): Sampling rate
        N (int): Window length
        H (int): Hop size
        norm_sum (bool): Normalizes by the number of summands in local autocorrelation (Default value = False)
        Theta (np.ndarray): Set of tempi (given in BPM) (Default value = np.arange(30, 601))

    Returns:
        tempogram (np.ndarray): Tempogram tempogram
        T_coef (np.ndarray): Time axis T_coef (seconds)
        F_coef_BPM (np.ndarray): Tempo axis F_coef_BPM (BPM)
        A_cut (np.ndarray): Time-lag representation A_cut (cut according to Theta)
        F_coef_lag_cut (np.ndarray): Lag axis F_coef_lag_cut
    """
    tempo_min = Theta[0]
    tempo_max = Theta[-1]
    lag_min = int(np.ceil(Fs * 60 / tempo_max))
    lag_max = int(np.ceil(Fs * 60 / tempo_min))
    A, T_coef, F_coef_lag = compute_autocorrelation_local(x, Fs, N, H, norm_sum=norm_sum)
    A_cut = A[lag_min:lag_max+1, :]
    F_coef_lag_cut = F_coef_lag[lag_min:lag_max+1]
    F_coef_BPM_cut = 60 / F_coef_lag_cut
    F_coef_BPM = Theta
    tempogram = interp1d(F_coef_BPM_cut, A_cut, kind='linear',
                         axis=0, fill_value='extrapolate')(F_coef_BPM)
    return tempogram, T_coef, F_coef_BPM, A_cut, F_coef_lag_cut


def compute_cyclic_tempogram(tempogram, F_coef_BPM, tempo_ref=30,
                             octave_bin=40, octave_num=4):
    """Compute cyclic tempogram

    Notebook: C6/C6S2_TempogramCyclic.ipynb

    Args:
        tempogram (np.ndarray): Input tempogram
        F_coef_BPM (np.ndarray): Tempo axis (BPM)
        tempo_ref (float): Reference tempo (BPM) (Default value = 30)
        octave_bin (int): Number of bins per tempo octave (Default value = 40)
        octave_num (int): Number of tempo octaves to be considered (Default value = 4)

    Returns:
        tempogram_cyclic (np.ndarray): Cyclic tempogram tempogram_cyclic
        F_coef_scale (np.ndarray): Tempo axis with regard to scaling parameter
        tempogram_log (np.ndarray): Tempogram with logarithmic tempo axis
        F_coef_BPM_log (np.ndarray): Logarithmic tempo axis (BPM)
    """
    F_coef_BPM_log = tempo_ref * np.power(2, np.arange(0, octave_num*octave_bin)/octave_bin)
    F_coef_scale = np.power(2, np.arange(0, octave_bin)/octave_bin)
    tempogram_log = interp1d(F_coef_BPM, tempogram, kind='linear', axis=0, fill_value='extrapolate')(F_coef_BPM_log)
    K = len(F_coef_BPM_log)
    tempogram_cyclic = np.zeros((octave_bin, tempogram.shape[1]))
    for m in np.arange(octave_bin):
        tempogram_cyclic[m, :] = np.mean(tempogram_log[m:K:octave_bin, :], axis=0)
    return tempogram_cyclic, F_coef_scale, tempogram_log, F_coef_BPM_log


def set_yticks_tempogram_cyclic(ax, octave_bin, F_coef_scale, num_tick=5):
    """Set yticks with regard to scaling parmater

    Notebook: C6/C6S2_TempogramCyclic.ipynb

    Args:
        ax (mpl.axes.Axes): Figure axis
        octave_bin (int): Number of bins per tempo octave
        F_coef_scale (np.ndarra): Tempo axis with regard to scaling parameter
        num_tick (int): Number of yticks (Default value = 5)
    """
    yticks = np.arange(0, octave_bin, octave_bin // num_tick)
    ax.set_yticks(yticks)
    ax.set_yticklabels(F_coef_scale[yticks].astype((np.unicode_, 4)))


@jit(nopython=True)
def compute_plp(X, Fs, L, N, H, Theta):
    """Compute windowed sinusoid with optimal phase

    Notebook: C6/C6S3_PredominantLocalPulse.ipynb

    Args:
        X (np.ndarray): Fourier-based (complex-valued) tempogram
        Fs (scalar): Sampling rate
        L (int): Length of novelty curve
        N (int): Window length
        H (int): Hop size
        Theta (np.ndarray): Set of tempi (given in BPM)

    Returns:
        nov_PLP (np.ndarray): PLP function
    """
    win = np.hanning(N)
    N_left = N // 2
    L_left = N_left
    L_right = N_left
    L_pad = L + L_left + L_right
    nov_PLP = np.zeros(L_pad)
    M = X.shape[1]
    tempogram = np.abs(X)
    for n in range(M):
        k = np.argmax(tempogram[:, n])
        tempo = Theta[k]
        omega = (tempo / 60) / Fs
        c = X[k, n]
        phase = - np.angle(c) / (2 * np.pi)
        t_0 = n * H
        t_1 = t_0 + N
        t_kernel = np.arange(t_0, t_1)
        kernel = win * np.cos(2 * np.pi * (t_kernel * omega - phase))
        nov_PLP[t_kernel] = nov_PLP[t_kernel] + kernel
    nov_PLP = nov_PLP[L_left:L_pad-L_right]
    nov_PLP[nov_PLP < 0] = 0
    return nov_PLP


def compute_plot_tempogram_plp(fn_wav, Fs=22050, N=500, H=10, Theta=np.arange(30, 601),
                               title='', figsize=(8, 4), plot_maxtempo=False):
    """Compute and plot Fourier-based tempogram and PLP function

    Notebook: C6/C6S3_PredominantLocalPulse.ipynb

    Args:
        fn_wav: Filename of audio file
        Fs: Sample rate (Default value = 22050)
        N: Window size (Default value = 500)
        H: Hop size (Default value = 10)
        Theta: Set of tempi (given in BPM) (Default value = np.arange(30, 601))
        title: Title of figure (Default value = '')
        figsize: Figure size (Default value = (8, 4))
        plot_maxtempo: Visualize tempo with greatest coefficients in tempogram (Default value = False)
    """
    x, Fs = librosa.load(fn_wav, Fs)

    nov, Fs_nov = libfmp.c6.compute_novelty_spectrum(x, Fs=Fs, N=2048, H=512, gamma=100, M=10, norm=True)
    nov, Fs_nov = libfmp.c6.resample_signal(nov, Fs_in=Fs_nov, Fs_out=100)

    L = len(nov)
    H = 10
    X, T_coef, F_coef_BPM = libfmp.c6.compute_tempogram_fourier(nov, Fs=Fs_nov, N=N, H=H, Theta=Theta)
    nov_PLP = compute_plp(X, Fs_nov, L, N, H, Theta)
    tempogram = np.abs(X)

    fig, ax = plt.subplots(2, 2, gridspec_kw={'width_ratios': [1, 0.05],
                                              'height_ratios': [2, 1]},
                           figsize=figsize)
    libfmp.b.plot_matrix(tempogram, T_coef=T_coef, F_coef=F_coef_BPM, title=title,
                         ax=[ax[0, 0], ax[0, 1]], ylabel='Tempo (BPM)', colorbar=True)
    if plot_maxtempo:
        coef_k = np.argmax(tempogram, axis=0)
        ax[0, 0].plot(T_coef, F_coef_BPM[coef_k], 'r.')

    t_nov = np.arange(nov.shape[0]) / Fs_nov
    peaks, properties = signal.find_peaks(nov_PLP, prominence=0.05)
    peaks_sec = t_nov[peaks]
    libfmp.b.plot_signal(nov_PLP, Fs_nov, color='k', ax=ax[1, 0])
    ax[1, 1].set_axis_off()
    ax[1, 0].plot(peaks_sec, nov_PLP[peaks], 'ro')
    plt.show()
    x_peaks = librosa.clicks(peaks_sec, sr=Fs, click_freq=1000, length=len(x))
    ipd.display(ipd.Audio(x + x_peaks, rate=Fs))
