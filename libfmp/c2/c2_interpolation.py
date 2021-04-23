"""
Module: libfmp.c2.C2_interpolation
Author: Meinard Müller
License: The MIT license, https://opensource.org/licenses/MIT

This file is part of the FMP Notebooks (https://www.audiolabs-erlangen.de/FMP)
"""
import numpy as np
from scipy.interpolate import interp1d


def compute_f_coef_linear(N, Fs, rho=1):
    """Refines the frequency vector by factor of rho

    Notebook: C2/C2_STFT-FreqGridInterpol.ipynb

    Args:
        N (int): Window size
        Fs (scalar): Sampling rate
        rho (int): Factor for refinement (Default value = 1)

    Returns:
        F_coef_new (np.ndarray): Refined frequency vector
    """
    L = rho * N
    F_coef_new = np.arange(0, L//2+1) * Fs / L
    return F_coef_new


def compute_f_coef_log(R, F_min, F_max):
    """Adapts the frequency vector in a logarithmic fashion

    Notebook: C2/C2_STFT-FreqGridInterpol.ipynb

    Args:
        R (scalar): Resolution (cents)
        F_min (float): Minimum frequency
        F_max (float): Maximum frequency (not included)

    Returns:
        F_coef_log (np.ndarray): Refined frequency vector with values given in Hz)
        F_coef_cents (np.ndarray): Refined frequency vector with values given in cents.
            Note: F_min serves as reference (0 cents)
    """
    n_bins = np.ceil(1200 * np.log2(F_max / F_min) / R).astype(int)
    F_coef_log = 2 ** (np.arange(0, n_bins) * R / 1200) * F_min
    F_coef_cents = 1200 * np.log2(F_coef_log / F_min)
    return F_coef_log, F_coef_cents


def interpolate_freq_stft(Y, F_coef, F_coef_new):
    """Interpolation of STFT along frequency axis

    Notebook: C2/C2_STFT-FreqGridInterpol.ipynb

    Args:
        Y (np.ndarray): Magnitude STFT
        F_coef (np.ndarray): Vector of frequency values
        F_coef_new (np.ndarray): Vector of new frequency values

    Returns:
        Y_interpol (np.ndarray): Interploated magnitude STFT
    """
    compute_Y_interpol = interp1d(F_coef, Y, kind='cubic', axis=0)
    Y_interpol = compute_Y_interpol(F_coef_new)
    return Y_interpol
