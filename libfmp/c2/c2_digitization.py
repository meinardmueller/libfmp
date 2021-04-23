"""
Module: libfmp.c2.c2_digitization
Author: Meinard Müller, Michael Krause
License: The MIT license, https://opensource.org/licenses/MIT

This file is part of the FMP Notebooks (https://www.audiolabs-erlangen.de/FMP)
"""
import numpy as np
from scipy.interpolate import interp1d
from matplotlib import pyplot as plt


def generate_function(Fs, dur=1):
    """Generate example function

    Notebook: C2/C2S2_DigitalSignalSampling.ipynb

    Args:
        Fs (scalar): Sampling rate
        dur (float): Duration (in seconds) of signal to be generated (Default value = 1)

    Returns:
        x (np.ndarray): Signal
        t (np.ndarray): Time axis (in seconds)
    """
    N = int(Fs * dur)
    t = np.arange(N) / Fs
    x = 1 * np.sin(2 * np.pi * (2 * t - 0))
    x += 0.5 * np.sin(2 * np.pi * (6 * t - 0.1))
    x += 0.1 * np.sin(2 * np.pi * (20 * t - 0.2))
    return x, t


def sampling_equidistant(x_1, t_1, Fs_2, dur=None):
    """Equidistant sampling of interpolated signal

    Notebook: C2/C2S2_DigitalSignalSampling.ipynb

    Args:
        x_1 (np.ndarray): Signal to be interpolated and sampled
        t_1 (np.ndarray): Time axis (in seconds) of x_1
        Fs_2 (scalar): Sampling rate used for equidistant sampling
        dur (float): Duration (in seconds) of sampled signal (Default value = None)

    Returns:
        x (np.ndarray): Sampled signal
        t (np.ndarray): Time axis (in seconds) of sampled signal
    """
    if dur is None:
        dur = len(t_1) * t_1[1]
    N = int(Fs_2 * dur)
    t_2 = np.arange(N) / Fs_2
    x_2 = interp1d(t_1, x_1, kind='linear', fill_value='extrapolate')(t_2)
    return x_2, t_2


def reconstruction_sinc(x, t, t_sinc):
    """Reconstruction from sampled signal using sinc-functions

    Notebook: C2/C2S2_DigitalSignalSampling.ipynb

    Args:
        x (np.ndarray): Sampled signal
        t (np.ndarray): Equidistant discrete time axis (in seconds) of x
        t_sinc (np.ndarray): Equidistant discrete time axis (in seconds) of signal to be reconstructed

    Returns:
        x_sinc (np.ndarray): Reconstructed signal having time axis t_sinc
    """
    Fs = 1 / t[1]
    x_sinc = np.zeros(len(t_sinc))
    for n in range(0, len(t)):
        x_sinc += x[n] * np.sinc(Fs * t_sinc - n)
    return x_sinc


def quantize_uniform(x, quant_min=-1.0, quant_max=1.0, quant_level=5):
    """Uniform quantization approach

    Notebook: C2/C2S2_DigitalSignalQuantization.ipynb

    Args:
        x (np.ndarray): Original signal
        quant_min (float): Minimum quantization level (Default value = -1.0)
        quant_max (float): Maximum quantization level (Default value = 1.0)
        quant_level (int): Number of quantization levels (Default value = 5)

    Returns:
        x_quant (np.ndarray): Quantized signal
    """
    x_normalize = (x-quant_min) * (quant_level-1) / (quant_max-quant_min)
    x_normalize[x_normalize > quant_level - 1] = quant_level - 1
    x_normalize[x_normalize < 0] = 0
    x_normalize_quant = np.around(x_normalize)
    x_quant = (x_normalize_quant) * (quant_max-quant_min) / (quant_level-1) + quant_min
    return x_quant


def plot_graph_quant_function(ax, quant_min=-1.0, quant_max=1.0, quant_level=256, mu=255.0, quant='uniform'):
    """Helper function for plotting a graph of quantization function and quantization error

    Notebook: C2/C2S2_DigitalSignalQuantization.ipynb

    Args:
        ax (mpl.axes.Axes): Axis
        quant_min (float): Minimum quantization level (Default value = -1.0)
        quant_max (float): Maximum quantization level (Default value = 1.0)
        quant_level (int): Number of quantization levels (Default value = 256)
        mu (float): Encoding parameter (Default value = 255.0)
        quant (str): Type of quantization (Default value = 'uniform')
    """
    x = np.linspace(quant_min, quant_max, 1000)
    if quant == 'uniform':
        x_quant = quantize_uniform(x, quant_min=quant_min, quant_max=quant_max, quant_level=quant_level)
        quant_stepsize = (quant_max - quant_min) / (quant_level-1)
        title = r'$\lambda = %d, \Delta=%0.2f$' % (quant_level, quant_stepsize)
    if quant == 'nonuniform':
        x_quant = quantize_nonuniform_mu(x, mu=mu, quant_level=quant_level)
        title = r'$\lambda = %d, \mu=%0.1f$' % (quant_level, mu)
    error = np.abs(x_quant - x)
    ax.plot(x, x, color='k', label='Original amplitude')
    ax.plot(x, x_quant, color='b', label='Quantized amplitude')
    ax.plot(x, error, 'r--', label='Quantization error')
    ax.set_title(title)
    ax.set_xlabel('Amplitude')
    ax.set_ylabel('Quantized amplitude/error')
    ax.set_xlim([quant_min, quant_max])
    ax.set_ylim([quant_min, quant_max])
    ax.grid('on')
    ax.legend()


def plot_signal_quant(x, t, x_quant, figsize=(8, 2), xlim=None, ylim=None, title=''):
    """Helper function for plotting a signal and its quantized version

    Notebook: C2/C2S2_DigitalSignalQuantization.ipynb

    Args:
        x: Original Signal
        t: Time
        x_quant: Quantized signal
        figsize: Figure size (Default value = (8, 2))
        xlim: Limits for x-axis (Default value = None)
        ylim: Limits for y-axis (Default value = None)
        title: Title of figure (Default value = '')
    """
    plt.figure(figsize=figsize)
    plt.plot(t, x, color='gray', linewidth=1.0, linestyle='-', label='Original signal')
    plt.plot(t, x_quant, color='red', linewidth=2.0, linestyle='-', label='Quantized signal')
    if xlim is None:
        plt.xlim([0, t[-1]])
    else:
        plt.xlim(xlim)
    if ylim is not None:
        plt.ylim(ylim)
    plt.xlabel('Time (seconds)')
    plt.ylabel('Amplitude')
    plt.title(title)
    plt.legend(loc='upper right', framealpha=1)
    plt.tight_layout()
    plt.show()


def encoding_mu_law(v, mu=255.0):
    """mu-law encoding

    Notebook: C2/C2S2_DigitalSignalQuantization.ipynb

    Args:
        v (float): Value between -1 and 1
        mu (float): Encoding parameter (Default value = 255.0)

    Returns:
        v_encode (float): Encoded value
    """
    v_encode = np.sign(v) * (np.log(1.0 + mu * np.abs(v)) / np.log(1.0 + mu))
    return v_encode


def decoding_mu_law(v, mu=255.0):
    """mu-law decoding

    Notebook: C2/C2S2_DigitalSignalQuantization.ipynb

    Args:
        v (float): Value between -1 and 1
        mu (float): Dencoding parameter (Default value = 255.0)

    Returns:
        v_decode (float): Decoded value
    """
    v_decode = np.sign(v) * (1.0 / mu) * ((1.0 + mu)**np.abs(v) - 1.0)
    return v_decode


def plot_mu_law(mu=255.0, figsize=(8.5, 4)):
    """Helper function for plotting a signal and its quantized version

    Notebook: C2/C2S2_DigitalSignalQuantization.ipynb

    Args:
        mu (float): Dencoding parameter (Default value = 255.0)
        figsize (tuple): Figure size (Default value = (8.5, 2))
    """
    values = np.linspace(-1, 1, 1000)
    values_encoded = encoding_mu_law(values, mu=mu)
    values_decoded = encoding_mu_law(values, mu=mu)

    plt.figure(figsize=figsize)
    ax = plt.subplot(1, 2, 1)
    ax.plot(values, values, color='k', label='Original values')
    ax.plot(values, values_encoded, color='b', label='Encoded values')
    ax.set_title(r'$\mu$-law encoding with $\mu=%.0f$' % mu)
    ax.set_xlabel('$v$')
    ax.set_ylabel(r'$F_\mu(v)$')
    ax.set_xlim([-1, 1])
    ax.set_ylim([-1, 1])
    ax.grid('on')
    ax.legend()

    ax = plt.subplot(1, 2, 2)
    ax.plot(values, values, color='k', label='Original values')
    ax.plot(values, values_decoded, color='b', label='Decoded values')
    ax.set_title(r'$\mu$-law decoding with $\mu=%.0f$' % mu)
    ax.set_xlabel('$v$')
    ax.set_ylabel(r'$F_\mu^{-1}(v)$')
    ax.set_xlim([-1, 1])
    ax.set_ylim([-1, 1])
    ax.grid('on')
    ax.legend()

    plt.tight_layout()
    plt.show()


def quantize_nonuniform_mu(x, mu=255.0, quant_level=256):
    """Nonuniform quantization approach using mu-encoding

    Notebook: C2/C2S2_DigitalSignalQuantization.ipynb

    Args:
        x (np.ndarray): Original signal
        mu (float): Encoding parameter (Default value = 255.0)
        quant_level (int): Number of quantization levels (Default value = 256)

    Returns:
        x_quant (np.ndarray): Quantized signal
    """
    x_en = encoding_mu_law(x, mu=mu)
    x_en_quant = quantize_uniform(x_en, quant_min=-1, quant_max=1, quant_level=quant_level)
    x_quant = decoding_mu_law(x_en_quant, mu=mu)
    return x_quant
