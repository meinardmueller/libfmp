"""
Module: libfmp.c8.c8s1_hps
Author: Meinard Müller, Frank Zalkow
License: The MIT license, https://opensource.org/licenses/MIT

This file is part of the FMP Notebooks (https://www.audiolabs-erlangen.de/FMP)
"""

from collections import OrderedDict
import numpy as np
from scipy import signal
import librosa
import IPython.display as ipd
import pandas as pd


def median_filter_horizontal(x, filter_len):
    """Apply median filter in horizontal direction

    Notebook: C8/C8S1_HPS.ipynb

    Args:
        x (np.ndarray): Input matrix
        filter_len (int): Filter length

    Returns:
        x_h (np.ndarray): Filtered matrix
    """
    return signal.medfilt(x, [1, filter_len])


def median_filter_vertical(x, filter_len):
    """Apply median filter in vertical direction

    Notebook: C8/C8S1_HPS.ipynb

    Args:
        x: Input matrix
        filter_len (int): Filter length

    Returns:
        x_p (np.ndarray): Filtered matrix
    """
    return signal.medfilt(x, [filter_len, 1])


def convert_l_sec_to_frames(L_h_sec, Fs=22050, N=1024, H=512):
    """Convert filter length parameter from seconds to frame indices

    Notebook: C8/C8S1_HPS.ipynb

    Args:
        L_h_sec (float): Filter length (in seconds)
        Fs (scalar): Sample rate (Default value = 22050)
        N (int): Window size (Default value = 1024)
        H (int): Hop size (Default value = 512)

    Returns:
        L_h (int): Filter length (in samples)
    """
    L_h = int(np.ceil(L_h_sec * Fs / H))
    return L_h


def convert_l_hertz_to_bins(L_p_Hz, Fs=22050, N=1024, H=512):
    """Convert filter length parameter from Hertz to frequency bins

    Notebook: C8/C8S1_HPS.ipynb

    Args:
        L_p_Hz (float): Filter length (in Hertz)
        Fs (scalar): Sample rate (Default value = 22050)
        N (int): Window size (Default value = 1024)
        H (int): Hop size (Default value = 512)

    Returns:
        L_p (int): Filter length (in frequency bins)
    """
    L_p = int(np.ceil(L_p_Hz * N / Fs))
    return L_p


def make_integer_odd(n):
    """Convert integer into odd integer

    Notebook: C8/C8S1_HPS.ipynb

    Args:
        n (int): Integer

    Returns:
        n (int): Odd integer
    """
    if n % 2 == 0:
        n += 1
    return n


def hps(x, Fs, N, H, L_h, L_p, L_unit='physical', mask='binary', eps=0.001, detail=False):
    """Harmonic-percussive separation (HPS) algorithm

    Notebook: C8/C8S1_HPS.ipynb

    Args:
        x (np.ndarray): Input signal
        Fs (scalar): Sampling rate of x
        N (int): Frame length
        H (int): Hopsize
        L_h (float): Horizontal median filter length given in seconds or frames
        L_p (float): Percussive median filter length given in Hertz or bins
        L_unit (str): Adjusts unit, either 'pyhsical' or 'indices' (Default value = 'physical')
        mask (str): Either 'binary' or 'soft' (Default value = 'binary')
        eps (float): Parameter used in soft maskig (Default value = 0.001)
        detail (bool): Returns detailed information (Default value = False)

    Returns:
        x_h (np.ndarray): Harmonic signal
        x_p (np.ndarray): Percussive signal
        details (dict): Dictionary containing detailed information; returned if ``detail=True``
    """
    assert L_unit in ['physical', 'indices']
    assert mask in ['binary', 'soft']
    # stft
    X = librosa.stft(x, n_fft=N, hop_length=H, win_length=N, window='hann', center=True, pad_mode='constant')
    # power spectrogram
    Y = np.abs(X) ** 2
    # median filtering
    if L_unit == 'physical':
        L_h = convert_l_sec_to_frames(L_h_sec=L_h, Fs=Fs, N=N, H=H)
        L_p = convert_l_hertz_to_bins(L_p_Hz=L_p, Fs=Fs, N=N, H=H)
    L_h = make_integer_odd(L_h)
    L_p = make_integer_odd(L_p)
    Y_h = signal.medfilt(Y, [1, L_h])
    Y_p = signal.medfilt(Y, [L_p, 1])

    # masking
    if mask == 'binary':
        M_h = np.int8(Y_h >= Y_p)
        M_p = np.int8(Y_h < Y_p)
    if mask == 'soft':
        eps = 0.00001
        M_h = (Y_h + eps / 2) / (Y_h + Y_p + eps)
        M_p = (Y_p + eps / 2) / (Y_h + Y_p + eps)
    X_h = X * M_h
    X_p = X * M_p

    # istft
    x_h = librosa.istft(X_h, hop_length=H, win_length=N, window='hann', center=True, length=x.size)
    x_p = librosa.istft(X_p, hop_length=H, win_length=N, window='hann', center=True, length=x.size)

    if detail:
        return x_h, x_p, dict(Y_h=Y_h, Y_p=Y_p, M_h=M_h, M_p=M_p, X_h=X_h, X_p=X_p)
    else:
        return x_h, x_p


def generate_audio_tag_html_list(list_x, Fs, width='150', height='40'):
    """Generates audio tag for html needed to be shown in table

    Notebook: C8/C8S1_HPS.ipynb

    Args:
        list_x (list): List of waveforms
        Fs (scalar): Sample rate
        width (str): Width in px (Default value = '150')
        height (str): Height in px (Default value = '40')

    Returns:
        audio_tag_html_list (list): List of HTML strings with audio tags
    """
    audio_tag_html_list = []
    for i in range(len(list_x)):
        audio_tag = ipd.Audio(list_x[i], rate=Fs)
        audio_tag_html = audio_tag._repr_html_().replace('\n', '').strip()
        audio_tag_html = audio_tag_html.replace('<audio ',
                                                '<audio style="width: '+width+'px; height: '+height+'px;"')
        audio_tag_html_list.append(audio_tag_html)
    return audio_tag_html_list


def hrps(x, Fs, N, H, L_h, L_p, beta=2.0, L_unit='physical', detail=False):
    """Harmonic-residual-percussive separation (HRPS) algorithm

    Notebook: C8/C8S1_HRPS.ipynb

    Args:
        x (np.ndarray): Input signal
        Fs (scalar): Sampling rate of x
        N (int): Frame length
        H (int): Hopsize
        L_h (float): Horizontal median filter length given in seconds or frames
        L_p (float): Percussive median filter length given in Hertz or bins
        beta (float): Separation factor (Default value = 2.0)
        L_unit (str): Adjusts unit, either 'pyhsical' or 'indices' (Default value = 'physical')
        detail (bool): Returns detailed information (Default value = False)

    Returns:
        x_h (np.ndarray): Harmonic signal
        x_p (np.ndarray): Percussive signal
        x_r (np.ndarray): Residual signal
        details (dict): Dictionary containing detailed information; returned if "detail=True"
    """
    assert L_unit in ['physical', 'indices']
    # stft
    X = librosa.stft(x, n_fft=N, hop_length=H, win_length=N, window='hann', center=True, pad_mode='constant')
    # power spectrogram
    Y = np.abs(X) ** 2
    # median filtering
    if L_unit == 'physical':
        L_h = convert_l_sec_to_frames(L_h_sec=L_h, Fs=Fs, N=N, H=H)
        L_p = convert_l_hertz_to_bins(L_p_Hz=L_p, Fs=Fs, N=N, H=H)
    L_h = make_integer_odd(L_h)
    L_p = make_integer_odd(L_p)
    Y_h = signal.medfilt(Y, [1, L_h])
    Y_p = signal.medfilt(Y, [L_p, 1])

    # masking
    M_h = np.int8(Y_h >= beta * Y_p)
    M_p = np.int8(Y_p > beta * Y_h)
    M_r = 1 - (M_h + M_p)
    X_h = X * M_h
    X_p = X * M_p
    X_r = X * M_r

    # istft
    x_h = librosa.istft(X_h, hop_length=H, win_length=N, window='hann', center=True, length=x.size)
    x_p = librosa.istft(X_p, hop_length=H, win_length=N, window='hann', center=True, length=x.size)
    x_r = librosa.istft(X_r, hop_length=H, win_length=N, window='hann', center=True, length=x.size)

    if detail:
        return x_h, x_p, x_r, dict(Y_h=Y_h, Y_p=Y_p, M_h=M_h, M_r=M_r, M_p=M_p, X_h=X_h, X_r=X_r, X_p=X_p)
    else:
        return x_h, x_p, x_r


def experiment_hps_parameter(fn_wav, param_list):
    """Script for running an HPS experiment over a parameter list, such as ``[[1024, 256, 0.1, 100], ...]``

    Notebook: C8/C8S1_HPS.ipynb

    Args:
        fn_wav (str): Path to wave file
        param_list (list): List of parameters
    """
    Fs = 22050
    x, Fs = librosa.load(fn_wav, sr=Fs)

    list_x = []
    list_x_h = []
    list_x_p = []
    list_N = []
    list_H = []
    list_L_h_sec = []
    list_L_p_Hz = []
    list_L_h = []
    list_L_p = []

    for param in param_list:
        N, H, L_h_sec, L_p_Hz = param
        print('N=%4d, H=%4d, L_h_sec=%4.2f, L_p_Hz=%3.1f' % (N, H, L_h_sec, L_p_Hz))
        x_h, x_p = hps(x, Fs=Fs, N=N, H=H, L_h=L_h_sec, L_p=L_p_Hz)
        L_h = convert_l_sec_to_frames(L_h_sec=L_h_sec, Fs=Fs, N=N, H=H)
        L_p = convert_l_hertz_to_bins(L_p_Hz=L_p_Hz, Fs=Fs, N=N, H=H)
        list_x.append(x)
        list_x_h.append(x_h)
        list_x_p.append(x_p)
        list_N.append(N)
        list_H.append(H)
        list_L_h_sec.append(L_h_sec)
        list_L_p_Hz.append(L_p_Hz)
        list_L_h.append(L_h)
        list_L_p.append(L_p)

    html_x = generate_audio_tag_html_list(list_x, Fs=Fs)
    html_x_h = generate_audio_tag_html_list(list_x_h, Fs=Fs)
    html_x_p = generate_audio_tag_html_list(list_x_p, Fs=Fs)

    pd.options.display.float_format = '{:,.1f}'.format
    pd.set_option('display.max_colwidth', None)
    df = pd.DataFrame(OrderedDict([
        ('$N$', list_N),
        ('$H$', list_H),
        ('$L_h$ (sec)', list_L_h_sec),
        ('$L_p$ (Hz)', list_L_p_Hz),
        ('$L_h$', list_L_h),
        ('$L_p$', list_L_p),
        ('$x$', html_x),
        ('$x_h$', html_x_h),
        ('$x_p$', html_x_p)]))
    df.index = np.arange(1, len(df) + 1)
    ipd.display(ipd.HTML(df.to_html(escape=False, index=False)))


def experiment_hrps_parameter(fn_wav, param_list):
    """Script for running an HRPS experiment over a parameter list, such as ``[[1024, 256, 0.1, 100], ...]``

    Args:
        fn_wav (str): Path to wave file
        param_list (list): List of parameters
    """
    Fs = 22050
    x, Fs = librosa.load(fn_wav, sr=Fs)

    list_x = []
    list_x_h = []
    list_x_p = []
    list_x_r = []
    list_N = []
    list_H = []
    list_L_h_sec = []
    list_L_p_Hz = []
    list_L_h = []
    list_L_p = []
    list_beta = []

    for param in param_list:
        N, H, L_h_sec, L_p_Hz, beta = param
        print('N=%4d, H=%4d, L_h_sec=%4.2f, L_p_Hz=%3.1f, beta=%3.1f' % (N, H, L_h_sec, L_p_Hz, beta))
        x_h, x_p, x_r = hrps(x, Fs=Fs, N=1024, H=512, L_h=L_h_sec, L_p=L_p_Hz, beta=beta)
        L_h = convert_l_sec_to_frames(L_h_sec=L_h_sec, Fs=Fs, N=N, H=H)
        L_p = convert_l_hertz_to_bins(L_p_Hz=L_p_Hz, Fs=Fs, N=N, H=H)
        list_x.append(x)
        list_x_h.append(x_h)
        list_x_p.append(x_p)
        list_x_r.append(x_r)
        list_N.append(N)
        list_H.append(H)
        list_L_h_sec.append(L_h_sec)
        list_L_p_Hz.append(L_p_Hz)
        list_L_h.append(L_h)
        list_L_p.append(L_p)
        list_beta.append(beta)

    html_x = generate_audio_tag_html_list(list_x, Fs=Fs)
    html_x_h = generate_audio_tag_html_list(list_x_h, Fs=Fs)
    html_x_p = generate_audio_tag_html_list(list_x_p, Fs=Fs)
    html_x_r = generate_audio_tag_html_list(list_x_r, Fs=Fs)

    pd.options.display.float_format = '{:,.1f}'.format
    pd.set_option('display.max_colwidth', None)
    df = pd.DataFrame(OrderedDict([
        ('$N$', list_N),
        ('$H$', list_H),
        ('$L_h$ (sec)', list_L_h_sec),
        ('$L_p$ (Hz)', list_L_p_Hz),
        ('$L_h$', list_L_h),
        ('$L_p$', list_L_p),
        ('$\\beta$', list_beta),
        ('$x$', html_x),
        ('$x_h$', html_x_h),
        ('$x_r$', html_x_r),
        ('$x_p$', html_x_p)]))

    df.index = np.arange(1, len(df) + 1)
    ipd.display(ipd.HTML(df.to_html(escape=False, index=False)))
