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
        N: Window size
        Fs: Sampling rate
        rho: Factor for refinement

    Returns:
        F_coef_linear: Refined frequency vector
    """
    L = rho * N
    F_coef_new = np.arange(0, L//2+1) * Fs / L
    return F_coef_new


def compute_f_coef_log(R, F_min, F_max):
    """Adapts the frequency vector in a logarithmic fashion

    Notebook: C2/C2_STFT-FreqGridInterpol.ipynb

    Args:
        R: Resolution (cents)
        F_min: minimum frequency
        F_max: maximum frequency (not included)

    Returns:
        F_coef_log: Refined frequency vector with values given in Hz)
        F_coef_cents: Refined frequency vector with values given in cents
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
        Y: Magnitude STFT
        F_coef: Vector of frequency values
        F_coef_new: Vector of new frequency values

    Returns:
        Y_interpol: Interploated magnitude STFT
    """
    compute_Y_interpol = interp1d(F_coef, Y, kind='cubic', axis=0)
    Y_interpol = compute_Y_interpol(F_coef_new)
    return Y_interpol
