"""
Module: libfmp.c8.c8s2_salience
Author: Sebastian Rosenzweig, Meinard Müller
License: The MIT license, https://opensource.org/licenses/MIT

This file is part of the FMP Notebooks (https://www.audiolabs-erlangen.de/FMP)
"""

import numpy as np
import librosa
from scipy import ndimage
from numba import jit

import libfmp.b
import libfmp.c8


@jit(nopython=True)
def principal_argument(v):
    """Principal argument function

    | Notebook: C6/C6S1_NoveltyPhase.ipynb, see also
    | Notebook: C8/C8S2_InstantFreqEstimation.ipynb

    Args:
        v (float or np.ndarray): Value (or vector of values)

    Returns:
        w (float or np.ndarray): Principle value of v
    """
    w = np.mod(v + 0.5, 1) - 0.5
    return w


@jit(nopython=True)
def compute_if(X, Fs, N, H):
    """Instantenous frequency (IF) estamation

    | Notebook: C8/C8S2_InstantFreqEstimation.ipynb, see also
    | Notebook: C6/C6S1_NoveltyPhase.ipynb

    Args:
        X (np.ndarray): STFT
        Fs (scalar): Sampling rate
        N (int): Window size in samples
        H (int): Hop size in samples

    Returns:
        F_coef_IF (np.ndarray): Matrix of IF values
    """
    phi_1 = np.angle(X[:, 0:-1]) / (2 * np.pi)
    phi_2 = np.angle(X[:, 1:]) / (2 * np.pi)

    K = X.shape[0]
    index_k = np.arange(0, K).reshape(-1, 1)
    # Bin offset (FMP, Eq. (8.45))
    kappa = (N / H) * principal_argument(phi_2 - phi_1 - index_k * H / N)
    # Instantaneous frequencies (FMP, Eq. (8.44))
    F_coef_IF = (index_k + kappa) * Fs / N

    # Extend F_coef_IF by copying first column to match dimensions of X
    F_coef_IF = np.hstack((np.copy(F_coef_IF[:, 0]).reshape(-1, 1), F_coef_IF))

    return F_coef_IF


@jit(nopython=True)
def f_coef(k, Fs, N):
    """STFT center frequency

    Notebook: C8/C8S2_SalienceRepresentation.ipynb

    Args:
        k (int): Coefficient number
        Fs  (scalar): Sampling rate in Hz
        N (int): Window length in samples

    Returns:
        freq (float): STFT center frequency
    """
    return k * Fs / N


@jit(nopython=True)
def frequency_to_bin_index(F, R=10.0, F_ref=55.0):
    """| Binning function with variable frequency resolution
    | Note: Indexing starts with 0 (opposed to [FMP, Eq. (8.49)])

    Notebook: C8/C8S2_SalienceRepresentation.ipynb

    Args:
        F (float): Frequency in Hz
        R (float): Frequency resolution in cents (Default value = 10.0)
        F_ref (float): Reference frequency in Hz (Default value = 55.0)

    Returns:
        bin_index (int): Index for bin (starting with index 0)
    """
    bin_index = np.floor((1200 / R) * np.log2(F / F_ref) + 0.5).astype(np.int64)
    return bin_index


@jit(nopython=True)
def p_bin(b, freq, R=10.0, F_ref=55.0):
    """Computes binning mask [FMP, Eq. (8.50)]

    Notebook: C8/C8S2_SalienceRepresentation.ipynb

    Args:
        b (int): Bin index
        freq (float): Center frequency
        R (float): Frequency resolution in cents (Default value = 10.0)
        F_ref (float): Reference frequency in Hz (Default value = 55.0)

    Returns:
        mask (float): Binning mask
    """
    mask = frequency_to_bin_index(freq, R, F_ref) == b
    mask = mask.reshape(-1, 1)
    return mask


@jit(nopython=True)
def compute_y_lf_bin(Y, Fs, N, R=10.0, F_min=55.0, F_max=1760.0):
    """Log-frequency Spectrogram with variable frequency resolution using binning

    Notebook: C8/C8S2_SalienceRepresentation.ipynb

    Args:
        Y (np.ndarray): Magnitude spectrogram
        Fs (scalar): Sampling rate in Hz
        N (int): Window length in samples
        R (float): Frequency resolution in cents (Default value = 10.0)
        F_min (float): Lower frequency bound (reference frequency) (Default value = 55.0)
        F_max (float): Upper frequency bound (is included) (Default value = 1760.0)

    Returns:
        Y_LF_bin (np.ndarray): Binned log-frequency spectrogram
        F_coef_hertz (np.ndarray): Frequency axis in Hz
        F_coef_cents (np.ndarray): Frequency axis in cents
    """
    # [FMP, Eq. (8.51)]
    B = frequency_to_bin_index(np.array([F_max]), R, F_min)[0] + 1
    F_coef_hertz = 2 ** (np.arange(0, B) * R / 1200) * F_min
    F_coef_cents = np.arange(0, B*R, R)
    Y_LF_bin = np.zeros((B, Y.shape[1]))

    K = Y.shape[0]
    freq = f_coef(np.arange(0, K), Fs, N)
    freq_lim_idx = np.where(np.logical_and(freq >= F_min, freq <= F_max))[0]
    freq_lim = freq[freq_lim_idx]
    Y_lim = Y[freq_lim_idx, :]

    for b in range(B):
        coef_mask = p_bin(b, freq_lim, R, F_min)
        Y_LF_bin[b, :] = (Y_lim*coef_mask).sum(axis=0)
    return Y_LF_bin, F_coef_hertz, F_coef_cents


@jit(nopython=True)
def p_bin_if(b, F_coef_IF, R=10.0, F_ref=55.0):
    """Computes binning mask for instantaneous frequency binning [FMP, Eq. (8.52)]

    Notebook: C8/C8S2_SalienceRepresentation.ipynb

    Args:
        b (int): Bin index
        F_coef_IF (float): Instantaneous frequencies
        R (float): Frequency resolution in cents (Default value = 10.0)
        F_ref (float): Reference frequency in Hz (Default value = 55.0)

    Returns:
        mask (np.ndarray): Binning mask
    """
    mask = frequency_to_bin_index(F_coef_IF, R, F_ref) == b
    return mask


@jit(nopython=True)
def compute_y_lf_if_bin(X, Fs, N, H, R=10, F_min=55.0, F_max=1760.0, gamma=0.0):
    """Binned Log-frequency Spectrogram with variable frequency resolution based on instantaneous frequency

    Notebook: C8/C8S2_SalienceRepresentation.ipynb

    Args:
        X (np.ndarray): Complex spectrogram
        Fs (scalar): Sampling rate in Hz
        N (int): Window length in samples
        H (int): Hopsize in samples
        R (float): Frequency resolution in cents (Default value = 10)
        F_min (float): Lower frequency bound (reference frequency) (Default value = 55.0)
        F_max (float): Upper frequency bound (Default value = 1760.0)
        gamma (float): Logarithmic compression factor (Default value = 0.0)

    Returns:
        Y_LF_IF_bin (np.ndarray): Binned log-frequency spectrogram using instantaneous frequency
        F_coef_hertz (np.ndarray): Frequency axis in Hz
        F_coef_cents (np.ndarray): Frequency axis in cents
    """
    # Compute instantaneous frequencies
    F_coef_IF = libfmp.c8.compute_if(X, Fs, N, H)
    freq_lim_mask = np.logical_and(F_coef_IF >= F_min, F_coef_IF < F_max)
    F_coef_IF = F_coef_IF * freq_lim_mask

    # Initialize ouput array and compute frequency axis
    B = frequency_to_bin_index(np.array([F_max]), R, F_min)[0] + 1
    F_coef_hertz = 2 ** (np.arange(0, B) * R / 1200) * F_min
    F_coef_cents = np.arange(0, B*R, R)
    Y_LF_IF_bin = np.zeros((B, X.shape[1]))

    # Magnitude binning
    if gamma == 0:
        Y = np.abs(X) ** 2
    else:
        Y = np.log(1 + np.float32(gamma)*np.abs(X))
    for b in range(B):
        coef_mask = p_bin_if(b, F_coef_IF, R, F_min)

        Y_LF_IF_bin[b, :] = (Y * coef_mask).sum(axis=0)
    return Y_LF_IF_bin, F_coef_hertz, F_coef_cents


@jit(nopython=True)
def harmonic_summation(Y, num_harm=10, alpha=1.0):
    """Harmonic summation for spectrogram [FMP, Eq. (8.54)]

    Notebook: C8/C8S2_SalienceRepresentation.ipynb

    Args:
        Y (np.ndarray): Magnitude spectrogram
        num_harm (int): Number of harmonics (Default value = 10)
        alpha (float): Weighting parameter (Default value = 1.0)

    Returns:
        Y_HS (np.ndarray): Spectrogram after harmonic summation
    """
    Y_HS = np.zeros(Y.shape)
    Y_zero_pad = np.vstack((Y, np.zeros((Y.shape[0]*num_harm, Y.shape[1]))))
    K = Y.shape[0]
    for k in range(K):
        harm_idx = np.arange(1, num_harm+1)*(k)
        weights = alpha ** (np.arange(1, num_harm+1) - 1).reshape(-1, 1)
        Y_HS[k, :] = (Y_zero_pad[harm_idx, :] * weights).sum(axis=0)
    return Y_HS


@jit(nopython=True)
def harmonic_summation_lf(Y_LF_bin, R, num_harm=10, alpha=1.0):
    """Harmonic summation for log-frequency spectrogram [FMP, Eq. (8.55)]

    Notebook: C8/C8S2_SalienceRepresentation.ipynb

    Args:
        Y_LF_bin (np.ndarray): Log-frequency spectrogram
        R (float): Frequency resolution in cents
        num_harm (int): Number of harmonics (Default value = 10)
        alpha (float): Weighting parameter (Default value = 1.0)

    Returns:
        Y_LF_bin_HS (np.ndarray): Log-frequency spectrogram after harmonic summation
    """
    Y_LF_bin_HS = np.zeros(Y_LF_bin.shape)
    pad_len = int(np.floor(np.log2(num_harm) * 1200 / R))
    Y_LF_bin_zero_pad = np.vstack((Y_LF_bin, np.zeros((pad_len, Y_LF_bin.shape[1]))))
    B = Y_LF_bin.shape[0]
    for b in range(B):
        harmonics = np.arange(1, num_harm+1)
        harm_idx = b + np.floor(np.log2(harmonics) * 1200 / R).astype(np.int64)
        weights = alpha ** (np.arange(1, num_harm+1) - 1).reshape(-1, 1)
        Y_LF_bin_HS[b, :] = (Y_LF_bin_zero_pad[harm_idx, :] * weights).sum(axis=0)
    return Y_LF_bin_HS


def compute_salience_rep(x, Fs, N, H, R, F_min=55.0, F_max=1760.0, num_harm=10, freq_smooth_len=11, alpha=1.0,
                         gamma=0.0):
    """Salience representation [FMP, Eq. (8.56)]

    Notebook: C8/C8S2_SalienceRepresentation.ipynb

    Args:
        x (np.ndarray): Audio signal
        Fs (scalar): Sampling frequency
        N (int): Window length in samples
        H (int): Hopsize in samples
        R (float): Frequency resolution in cents
        F_min (float): Lower frequency bound (reference frequency) (Default value = 55.0)
        F_max (float): Upper frequency bound (Default value = 1760.0)
        num_harm (int): Number of harmonics (Default value = 10)
        freq_smooth_len (int): Filter length for vertical smoothing (Default value = 11)
        alpha (float): Weighting parameter (Default value = 1.0)
        gamma (float): Logarithmic compression factor (Default value = 0.0)

    Returns:
        Z (np.ndarray): Salience representation
        F_coef_hertz (np.ndarray): Frequency axis in Hz
        F_coef_cents (np.ndarray): Frequency axis in cents
    """
    X = librosa.stft(x, n_fft=N, hop_length=H, win_length=N, pad_mode='constant')
    Y_LF_IF_bin, F_coef_hertz, F_coef_cents = compute_y_lf_if_bin(X, Fs, N, H, R, F_min, F_max, gamma=gamma)
    # smoothing
    Y_LF_IF_bin = ndimage.filters.convolve1d(Y_LF_IF_bin, np.hanning(freq_smooth_len), axis=0, mode='constant')
    Z = harmonic_summation_lf(Y_LF_IF_bin, R=R, num_harm=num_harm, alpha=alpha)
    return Z, F_coef_hertz, F_coef_cents
