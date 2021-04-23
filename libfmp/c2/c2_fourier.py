"""
Module: libfmp.c2.c2_fourier
Author: Frank Zalkow, Meinard Müller
License: The MIT license, https://opensource.org/licenses/MIT

This file is part of the FMP Notebooks (https://www.audiolabs-erlangen.de/FMP)
"""
import numpy as np
from numba import jit
import librosa


@jit(nopython=True)
def generate_matrix_dft(N, K):
    """Generates a DFT (discrete Fourier transfrom) matrix

    Notebook: C2/C2_DFT-FFT.ipynb

    Args:
        N (int): Number of samples
        K (int): Number of frequency bins

    Returns:
        dft (np.ndarray): The DFT matrix
    """
    dft = np.zeros((K, N), dtype=np.complex128)
    for n in range(N):
        for k in range(K):
            dft[k, n] = np.exp(-2j * np.pi * k * n / N)
    return dft


@jit(nopython=True)
def generate_matrix_dft_inv(N, K):
    """Generates an IDFT (inverse discrete Fourier transfrom) matrix

    Notebook: C2/C2_STFT-Inverse.ipynb

    Args:
        N (int): Number of samples
        K (int): Number of frequency bins

    Returns:
        dft (np.ndarray): The IDFT matrix
    """
    dft = np.zeros((K, N), dtype=np.complex128)
    for n in range(N):
        for k in range(K):
            dft[k, n] = np.exp(2j * np.pi * k * n / N) / N
    return dft


@jit(nopython=True)
def dft(x):
    """Compute the disrcete Fourier transfrom (DFT)

    Notebook: C2/C2_DFT-FFT.ipynb

    Args:
        x (np.ndarray): Signal to be transformed

    Returns:
        X (np.ndarray): Fourier transform of x
    """
    x = x.astype(np.complex128)
    N = len(x)
    dft_mat = generate_matrix_dft(N, N)/N
    return np.dot(dft_mat, x)


@jit(nopython=True)
def idft(X):
    """Compute the inverse discrete Fourier transfrom (IDFT)

    Args:
        X (np.ndarray): Signal to be transformed

    Returns:
        x (np.ndarray): Inverse Fourier transform of X
    """
    X = X.astype(np.complex128)
    N = len(X)
    dft_mat = generate_matrix_dft_inv(N, N)
    return np.dot(dft_mat, X)


@jit(nopython=True)
def twiddle(N):
    """Generate the twiddle factors used in the computation of the fast Fourier transform (FFT)

    Notebook: C2/C2_DFT-FFT.ipynb

    Args:
        N (int): Number of samples

    Returns:
        sigma (np.ndarray): The twiddle factors
    """
    k = np.arange(N // 2)
    sigma = np.exp(-2j * np.pi * k / N)
    return sigma


@jit(nopython=True)
def twiddle_inv(N):
    """Generate the twiddle factors used in the computation of the Inverse fast Fourier transform (IFFT)

    Args:
        N (int): Number of samples

    Returns:
        sigma (np.ndarray): The twiddle factors
    """
    n = np.arange(N // 2)
    sigma = np.exp(2j * np.pi * n / N)
    return sigma


@jit(nopython=True)
def fft(x):
    """Compute the fast Fourier transform (FFT)

    Notebook: C2/C2_DFT-FFT.ipynb

    Args:
        x (np.ndarray): Signal to be transformed

    Returns:
        X (np.ndarray): Fourier transform of x
    """
    x = x.astype(np.complex128)
    N = len(x)
    log2N = np.log2(N)
    assert log2N == int(log2N), 'N must be a power of two!'
    X = np.zeros(N, dtype=np.complex128)

    if N == 1:
        return x
    else:
        this_range = np.arange(N)
        A = fft(x[this_range % 2 == 0])
        B = fft(x[this_range % 2 == 1])
        C = twiddle(N) * B
        X[:N//2] = A + C
        X[N//2:] = A - C
        return X


@jit(nopython=True)
def ifft_noscale(X):
    """Compute the inverse fast Fourier transform (IFFT) without the final scaling factor of 1/N

    Args:
        X (np.ndarray): Fourier transform of x

    Returns:
        x (np.ndarray): Inverse Fourier transform of X
    """
    X = X.astype(np.complex128)
    N = len(X)
    log2N = np.log2(N)
    assert log2N == int(log2N), 'N must be a power of two!'
    x = np.zeros(N, dtype=np.complex128)

    if N == 1:
        return X
    else:
        this_range = np.arange(N)
        A = ifft_noscale(X[this_range % 2 == 0])
        B = ifft_noscale(X[this_range % 2 == 1])
        C = twiddle_inv(N) * B
        x[:N//2] = A + C
        x[N//2:] = A - C
        return x


@jit(nopython=True)
def ifft(X):
    """Compute the inverse fast Fourier transform (IFFT)

    Args:
        X (np.ndarray): Fourier transform of x

    Returns:
        x (np.ndarray): Inverse Fourier transform of X
    """
    return ifft_noscale(X) / len(X)


def stft_basic(x, w, H=8, only_positive_frequencies=False):
    """Compute a basic version of the discrete short-time Fourier transform (STFT)

    Notebook: C2/C2_STFT-Basic.ipynb

    Args:
        x (np.ndarray): Signal to be transformed
        w (np.ndarray): Window function
        H (int): Hopsize (Default value = 8)
        only_positive_frequencies (bool): Return only positive frequency part of spectrum (non-invertible)
            (Default value = False)

    Returns:
        X (np.ndarray): The discrete short-time Fourier transform
    """
    N = len(w)
    L = len(x)
    M = np.floor((L - N) / H).astype(int) + 1
    X = np.zeros((N, M), dtype='complex')
    for m in range(M):
        x_win = x[m * H:m * H + N] * w
        X_win = np.fft.fft(x_win)
        X[:, m] = X_win

    if only_positive_frequencies:
        K = 1 + N // 2
        X = X[0:K, :]
    return X


def istft_basic(X, w, H, L):
    """Compute the inverse of the basic discrete short-time Fourier transform (ISTFT)

    Notebook: C2/C2_STFT-Inverse.ipynb

    Args:
        X (np.ndarray): The discrete short-time Fourier transform
        w (np.ndarray): Window function
        H (int): Hopsize
        L (int): Length of time signal

    Returns:
        x (np.ndarray): Time signal
    """
    N = len(w)
    M = X.shape[1]
    x_win_sum = np.zeros(L)
    w_sum = np.zeros(L)
    for m in range(M):
        x_win = np.fft.ifft(X[:, m])
        # Avoid imaginary values (due to floating point arithmetic)
        x_win = np.real(x_win)
        x_win_sum[m * H:m * H + N] = x_win_sum[m * H:m * H + N] + x_win
        w_shifted = np.zeros(L)
        w_shifted[m * H:m * H + N] = w
        w_sum = w_sum + w_shifted
    # Avoid division by zero
    w_sum[w_sum == 0] = np.finfo(np.float32).eps
    x_rec = x_win_sum / w_sum
    return x_rec, x_win_sum, w_sum


@jit(nopython=True)
def stft(x, w, H=512, zero_padding=0, only_positive_frequencies=False):
    """Compute the discrete short-time Fourier transform (STFT)

    Args:
        x (np.ndarray): Signal to be transformed
        w (np.ndarray): Window function
        H (int): Hopsize (Default value = 512)
        zero_padding (bool): Number of zeros to be padded after windowing and before the Fourier transform of a frame
            (Note: The purpose of this step is to increase the frequency sampling.) (Default value = 0)
        only_positive_frequencies (bool): Return only positive frequency part of spectrum (non-invertible)
            (Default value = False)

    Returns:
        X (np.ndarray): The discrete short-time Fourier transform
    """

    N = len(w)
    x = np.concatenate((np.zeros(N // 2), x, np.zeros(N // 2)))

    L = len(x)
    M = int(np.floor((L - N) / H)) + 1

    X = np.zeros((N + zero_padding, M), dtype=np.complex128)
    zero_padding_vector = np.zeros((zero_padding, ), dtype=x.dtype)

    for m in range(M):
        x_win = x[m * H:m * H + N] * w
        if zero_padding > 0:
            x_win = np.concatenate((x_win, zero_padding_vector))
        X_win = fft(x_win)
        # Note: X_win = np.fft.fft(x_win) does not work in combination with @jit
        X[:, m] = X_win

    if only_positive_frequencies:
        K = 1 + (N + zero_padding) // 2
        X = X[0:K, :]
    return X


@jit(nopython=True)
def istft(X, w, H, L, zero_padding=0):
    """Compute the inverse discrete short-time Fourier transform (ISTFT)

    Args:
        X (np.ndarray): The discrete short-time Fourier transform
        w (np.ndarray): Window function
        H (int): Hopsize
        L (int): Length of time signal
        zero_padding (bool): Number of zeros to be padded after windowing and before the Fourier transform of a frame
            (Default value = 0)

    Returns:
        x (np.ndarray): Reconstructed time signal
    """
    N = len(w)
    L = L + N
    M = X.shape[1]
    w_sum = np.zeros(L)
    x_win_sum = np.zeros(L)
    w_sum = np.zeros(L)
    for m in range(M):
        start_idx, end_idx = m * H, m * H + N + zero_padding
        if start_idx > L:
            break

        x_win = ifft(X[:, m])
        # Note: x_win = np.fft.ifft(X[:, m]) does not work in combination with @jit
        if end_idx > L:
            end_idx = L
            x_win = x_win[:end_idx-start_idx]
            cur_w = w[:end_idx-start_idx]
        else:
            cur_w = w

        # Avoid imaginary values (due to floating point arithmetic)
        x_win_real = np.real(x_win)
        x_win_sum[start_idx:end_idx] = x_win_sum[start_idx:end_idx] + x_win_real
        w_shifted = np.zeros(L)
        w_shifted[start_idx:start_idx + len(cur_w)] = cur_w
        w_sum = w_sum + w_shifted
    # Avoid division by zero
    w_sum[w_sum == 0] = np.finfo(np.float32).eps
    x_rec = x_win_sum / w_sum
    x_rec = x_rec[N // 2:-N // 2]
    return x_rec


def stft_convention_fmp(x, Fs, N, H, pad_mode='constant', center=True, mag=False, gamma=0):
    """Compute the discrete short-time Fourier transform (STFT)

    Notebook: C2/C2_STFT-FreqGridInterpol.ipynb

    Args:
        x (np.ndarray): Signal to be transformed
        Fs (scalar): Sampling rate
        N (int): Window size
        H (int): Hopsize
        pad_mode (str): Padding strategy is used in librosa (Default value = 'constant')
        center (bool): Centric view as used in librosa (Default value = True)
        mag (bool): Computes magnitude STFT if mag==True (Default value = False)
        gamma (float): Constant for logarithmic compression (only applied when mag==True) (Default value = 0)

    Returns:
        X (np.ndarray): Discrete (magnitude) short-time Fourier transform
    """
    X = librosa.stft(x, n_fft=N, hop_length=H, win_length=N,
                     window='hann', pad_mode=pad_mode, center=center)
    if mag:
        X = np.abs(X)**2
        if gamma > 0:
            X = np.log(1 + gamma * X)
    F_coef = librosa.fft_frequencies(sr=Fs, n_fft=N)
    T_coef = librosa.frames_to_time(np.arange(X.shape[1]), sr=Fs, hop_length=H)
    # T_coef = np.arange(X.shape[1]) * H/Fs
    # F_coef = np.arange(N//2+1) * Fs/N
    return X, T_coef, F_coef
