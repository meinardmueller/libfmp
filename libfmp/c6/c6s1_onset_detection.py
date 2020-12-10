"""
Module: libfmp.c6.c6s1_onset_detection
Author: Meinard Müller, Angel Villar-Corrales
License: The MIT license, https://opensource.org/licenses/MIT

This file is part of the FMP Notebooks (https://www.audiolabs-erlangen.de/FMP)
"""

import numpy as np
from numba import jit
from scipy import signal
from scipy.interpolate import interp1d
from scipy import ndimage
import librosa
import libfmp.b


def read_annotation_pos(fn_ann, label='', header=1, print_table=False):
    """Read and convert file containing either list of pairs (number,label) or list of (number)

    Notebook: C6/C6S1_OnsetDetection.ipynb

    Args:
        fn_ann: Name of file
        label: Name of label
        header: Assumes header (1) or not (0)
        print_table: Prints table if True

    Returns:
        ann: List Annotations
        label_keys: Dictionaries specifying color and line style used for labels
    """
    df = libfmp.b.read_csv(fn_ann, header=header)
    if print_table:
        print(df)
    num_col = df.values[0].shape[0]
    if num_col == 1:
        df = df.assign(label=[label] * len(df.index))
    ann = df.values.tolist()

    label_keys = {'beat': {'linewidth': 2, 'linestyle': ':', 'color': 'r'},
                  'onset': {'linewidth': 1, 'linestyle': ':', 'color': 'r'}}
    return ann, label_keys


def compute_novelty_energy(x, Fs=1, N=2048, H=128, gamma=10, norm=1):
    """Compute energy-based novelty function

    Notebook: C6/C6S1_NoveltyEnergy.ipynb

    Args:
        x: Signal
        Fs: Sampling rate
        N: Window size
        H: Hope size
        gamma: Parameter for logarithmic compression
        norm: Apply max norm (if norm==1)

    Returns:
        novelty_energy: Energy-based novelty function
        Fs_feature: Feature rate
    """
    #x_power = x**2
    w = signal.hann(N)
    Fs_feature = Fs/H
    energy_local = np.convolve(x**2, w**2, 'same')
    energy_local = energy_local[::H]
    if gamma is not None:
        energy_local = np.log(1 + gamma * energy_local)
    energy_local_diff = np.diff(energy_local)
    energy_local_diff = np.concatenate((energy_local_diff, np.array([0])))
    novelty_energy = np.copy(energy_local_diff)
    novelty_energy[energy_local_diff < 0] = 0
    if norm == 1:
        max_value = max(novelty_energy)
        if max_value > 0:
            novelty_energy = novelty_energy / max_value
    return novelty_energy, Fs_feature


@jit(nopython=True)
def compute_local_average(x, M):
    """Compute local average of signal

    Notebook: C6/C6S1_NoveltySpectral.ipynb

    Args:
        x: Signal
        M: Determines size (2M+1) in samples of centric window  used for local average

    Returns:
        local_average: Local average signal
    """
    L = len(x)
    local_average = np.zeros(L)
    for m in range(L):
        a = max(m - M, 0)
        b = min(m + M + 1, L)
        local_average[m] = (1 / (2 * M + 1)) * np.sum(x[a:b])
    return local_average


def compute_novelty_spectrum(x, Fs=1, N=1024, H=256, gamma=100, M=10, norm=1):
    """Compute spectral-based novelty function

    Notebook: C6/C6S1_NoveltySpectral.ipynb

    Args:
        x: Signal
        Fs: Sampling rate
        N: Window size
        H: Hope size
        gamma: Parameter for logarithmic compression
        M: Size (frames) of local average
        norm: Apply max norm (if norm==1)

    Returns:
        novelty_spectrum: Energy-based novelty function
        Fs_feature: Feature rate
    """
    X = librosa.stft(x, n_fft=N, hop_length=H, win_length=N, window='hanning')
    Fs_feature = Fs / H
    Y = np.log(1 + gamma * np.abs(X))
    Y_diff = np.diff(Y)
    Y_diff[Y_diff < 0] = 0
    novelty_spectrum = np.sum(Y_diff, axis=0)
    novelty_spectrum = np.concatenate((novelty_spectrum, np.array([0.0])))
    if M > 0:
        local_average = compute_local_average(novelty_spectrum, M)
        novelty_spectrum = novelty_spectrum - local_average
        novelty_spectrum[novelty_spectrum < 0] = 0.0
    if norm == 1:
        max_value = max(novelty_spectrum)
        if max_value > 0:
            novelty_spectrum = novelty_spectrum / max_value
    return novelty_spectrum, Fs_feature


def principal_argument(v):
    """Principal argument function

    Notebook: /C6/C6S1_NoveltyPhase.ipynb, see also C8/C8S2_InstantFreqEstimation.ipynb

    Args:
        v: value (or vector of values)

    Returns:
        w: Principle value of v
    """
    w = np.mod(v + 0.5, 1) - 0.5
    return w


def compute_novelty_phase(x, Fs=1, N=1024, H=64, M=40, norm=1):
    """Compute phase-based novelty function

    Notebook: C6/C6/C6S1_NoveltyPhase.ipynb

    Args:
        x: Signal
        Fs: Sampling rate
        N: Window size
        H: Hop size
        M: Determines size (2M+1) in samples of centric window  used for local average
        norm: Apply max norm (if norm==1)

    Returns:
        novelty_spectrum: Energy-based novelty function
        Fs_feature: Feature rate
    """
    X = librosa.stft(x, n_fft=N, hop_length=H, win_length=N, window='hanning')
    Fs_feature = Fs / H
    phase = np.angle(X) / (2*np.pi)
    phase_diff = principal_argument(np.diff(phase, axis=1))
    phase_diff2 = principal_argument(np.diff(phase_diff, axis=1))
    novelty_phase = np.sum(np.abs(phase_diff2), axis=0)
    novelty_phase = np.concatenate((novelty_phase, np.array([0, 0])))
    if M > 0:
        local_average = compute_local_average(novelty_phase, M)
        novelty_phase = novelty_phase - local_average
        novelty_phase[novelty_phase < 0] = 0
    if norm == 1:
        max_value = np.max(novelty_phase)
        if max_value > 0:
            novelty_phase = novelty_phase / max_value
    return novelty_phase, Fs_feature


def compute_novelty_complex(x, Fs=1, N=1024, H=64, gamma=10, M=40, norm=1):
    """Compute complex-domain novelty function

    Notebook: C6/C6S1_NoveltyComplex.ipynb

    Args:
        x: Signal
        Fs: Sampling rate
        N: Window size
        H: Hop size
        M: Determines size (2M+1) in samples of centric window  used for local average
        norm: Apply max norm (if norm==1)

    Returns:
        novelty_spectrum: Energy-based novelty function
        Fs_feature: Feature rate
    """
    X = librosa.stft(x, n_fft=N, hop_length=H, win_length=N, window='hanning')
    Fs_feature = Fs / H
    mag = np.abs(X)
    if gamma > 0:
        mag = np.log(1 + gamma * mag)
    phase = np.angle(X) / (2*np.pi)
    phase_diff = np.diff(phase, axis=1)
    phase_diff = np.concatenate((phase_diff, np.zeros((phase.shape[0], 1))), axis=1)
    X_hat = mag * np.exp(2*np.pi*1j*(phase+phase_diff))
    X_prime = np.abs(X_hat - X)
    X_plus = np.copy(X_prime)
    for n in range(1, X.shape[0]):
        idx = np.where(mag[n, :] < mag[n-1, :])
        X_plus[n, idx] = 0
    novelty_complex = np.sum(X_plus, axis=0)
    if M > 0:
        local_average = compute_local_average(novelty_complex, M)
        novelty_complex = novelty_complex - local_average
        novelty_complex[novelty_complex < 0] = 0
    if norm == 1:
        max_value = np.max(novelty_complex)
        if max_value > 0:
            novelty_complex = novelty_complex / max_value
    return novelty_complex, Fs_feature


def resample_signal(x_in, Fs_in, Fs_out=100, norm=1, time_max_sec=None, sigma=None):
    """Resample and smooth signal

    Notebook: C6/C6S1_NoveltyComparison.ipynb

    Args:
        x_in: Input signal
        Fs_in: Sampling rate of input signal
        Fs_out: Sampling rate of output signal
        norm: Apply max norm (if norm==1)
        time_max_sec: Duration of output signal (given in seconds)
        sigma: Standard deviation for smoothing Gaussian kernel

    Returns:
        x_out: Output signal
        F_out: Feature rate of output signal
    """
    if sigma is not None:
        x_in = ndimage.gaussian_filter(x_in, sigma=sigma)
    T_coef_in = np.arange(x_in.shape[0]) / Fs_in
    time_in_max_sec = T_coef_in[-1]
    if time_max_sec is None:
        time_max_sec = time_in_max_sec
    N_out = int(np.ceil(time_max_sec*Fs_out))
    T_coef_out = np.arange(N_out) / Fs_out
    if T_coef_out[-1] > time_in_max_sec:
        x_in = np.append(x_in, [0])
        T_coef_in = np.append(T_coef_in, [T_coef_out[-1]])
    x_out = interp1d(T_coef_in, x_in, kind='linear')(T_coef_out)
    if norm == 1:
        x_max = max(x_out)
        if x_max > 0:
            x_out = x_out / max(x_out)
    return x_out, Fs_out


def average_nov_dic(nov_dic, time_max_sec, Fs_out=100, norm=1, sigma=None):
    """Average respamples set of novelty functions

    Notebook: C6/C6S1_NoveltyComparison.ipynb

    Args:
        nov_dic: Dictionary of novelty functions
        time_max_sec: Duration of output signals (given in seconds)
        Fs_out: Sampling rate of output signal
        norm: Apply max norm (if norm==1)
        sigma: Standard deviation for smoothing Gaussian kernel

    Returns:
        nov_matrix: Matrix containing resampled output signal (last one is average)
        Fs_out: Sampling rate of output signals
    """
    nov_num = len(nov_dic)
    N_out = int(np.ceil(time_max_sec*Fs_out))
    nov_matrix = np.zeros([nov_num + 1, N_out])
    for k in range(nov_num):
        nov = nov_dic[k][0]
        Fs_nov = nov_dic[k][1]
        nov_out, Fs_out = resample_signal(nov, Fs_in=Fs_nov, Fs_out=Fs_out,
                                          time_max_sec=time_max_sec, sigma=sigma)
        nov_matrix[k, :] = nov_out
    nov_average = np.sum(nov_matrix, axis=0)/nov_num
    if norm == 1:
        max_value = np.max(nov_average)
        if max_value > 0:
            nov_average = nov_average / max_value
    nov_matrix[k+1, :] = nov_average
    return nov_matrix, Fs_out

# #####################################################################
#
# def plot_novelty(novelty, Fs, ax=None, figsize=(6, 2), color='k',
#                  xlabel='Time (seconds)', ylabel='', title=''):
#     t = np.arange(novelty.shape[0]) / Fs
#     if ax==None:
#         axplot = plt.figure(figsize=figsize)
#         axplot = plt.subplot(1,1,1)
#     else:
#         axplot=ax
#     axplot.plot(t, novelty, color=color)
#     axplot.set_xlim([t[0], t[-1]])
#     axplot.set_ylim([1.1*np.min(novelty), 1.1*np.max(novelty)])
#     axplot.set_xlabel(xlabel)
#     axplot.set_ylabel(ylabel)
#     axplot.set_title(title)
#     if ax!=None: plt.tight_layout()
#
#
# def compute_novelty_resample(x, Fs=1, N=1024, H=256, Fs_out=100):
#     nov, Fs_nov = compute_novelty_spectrum(x, Fs=Fs, N=N, H=H)
#     T_coef_in = np.arange(nov.shape[0]) / Fs_nov
#     N_out = int(np.round(T_coef_in[-1]*Fs_out) + 1)
#     T_coef_out = np.arange(N_out) / Fs_out
#     nov_out = resample_signal(nov, T_coef_in, T_coef_out)
#     return nov_out, Fs_out
