"""
Module: libfmp.c1.c1s3_audio_rep
Author: Meinard Müller
License: The MIT license, https://opensource.org/licenses/MIT

This file is part of the FMP Notebooks (https://www.audiolabs-erlangen.de/FMP)
"""
import numpy as np
from matplotlib import pyplot as plt
import librosa
import IPython.display as ipd


def f_pitch(p):
    """Compute center frequency for (single or array of) MIDI note numbers

    Notebook: C1/C1S3_FrequencyPitch.ipynb

    Args:
        p (float or np.ndarray): MIDI note numbers

    Returns:
        freq_center (float or np.ndarray): Center frequency
    """
    freq_center = 2 ** ((p - 69) / 12) * 440
    return freq_center


def difference_cents(freq_1, freq_2):
    """Difference between two frequency values specified in cents

    Notebook: C1/C1S3_FrequencyPitch.ipynb

    Args:
        freq_1 (float): First frequency
        freq_2 (float): Second frequency

    Returns:
        delta (float): Difference in cents
    """
    delta = np.log2(freq_1 / freq_2) * 1200
    return delta


def generate_sinusoid(dur=5, Fs=1000, amp=1, freq=1, phase=0):
    """Generation of sinusoid

    Notebook: C1/C1S3_FrequencyPitch.ipynb

    Args:
        dur (float): Duration (in seconds) (Default value = 5)
        Fs (scalar): Sampling rate (Default value = 1000)
        amp (float): Amplitude of sinusoid (Default value = 1)
        freq (float): Frequency of sinusoid (Default value = 1)
        phase (float): Phase of sinusoid (Default value = 0)

    Returns:
        x (np.ndarray): Signal
        t (np.ndarray): Time axis (in seconds)

    """
    num_samples = int(Fs * dur)
    t = np.arange(num_samples) / Fs
    x = amp * np.sin(2*np.pi*(freq*t-phase))
    return x, t


def compute_power_db(x, Fs, win_len_sec=0.1, power_ref=10**(-12)):
    """Computation of the signal power in dB

    Notebook: C1/C1S3_Dynamics.ipynb

    Args:
        x (np.ndarray): Signal (waveform) to be analyzed
        Fs (scalar): Sampling rate
        win_len_sec (float): Length (seconds) of the window (Default value = 0.1)
        power_ref (float): Reference power level (0 dB) (Default value = 10**(-12))

    Returns:
        power_db (np.ndarray): Signal power in dB
    """
    win_len = round(win_len_sec * Fs)
    win = np.ones(win_len) / win_len
    power_db = 10 * np.log10(np.convolve(x**2, win, mode='same') / power_ref)
    return power_db


def compute_equal_loudness_contour(freq_min=30, freq_max=15000, num_points=100):
    """Computation of the equal loudness contour

    Notebook: C1/C1S3_Dynamics.ipynb

    Args:
        freq_min (float): Lowest frequency to be evaluated (Default value = 30)
        freq_max (float): Highest frequency to be evaluated (Default value = 15000)
        num_points (int): Number of evaluation points (Default value = 100)

    Returns:
        equal_loudness_contour (np.ndarray): Equal loudness contour (in dB)
        freq_range (np.ndarray): Evaluated frequency points
    """
    freq_range = np.logspace(np.log10(freq_min), np.log10(freq_max), num=num_points)
    freq = 1000
    # Function D from https://bar.wikipedia.org/wiki/Datei:Acoustic_weighting_curves.svg
    h_freq = ((1037918.48 - freq**2)**2 + 1080768.16 * freq**2) / ((9837328 - freq**2)**2 + 11723776 * freq**2)
    n_freq = (freq / (6.8966888496476 * 10**(-5))) * np.sqrt(h_freq / ((freq**2 + 79919.29) * (freq**2 + 1345600)))
    h_freq_range = ((1037918.48 - freq_range**2)**2 + 1080768.16 * freq_range**2) / ((9837328 - freq_range**2)**2
                                                                                     + 11723776 * freq_range**2)
    n_freq_range = (freq_range / (6.8966888496476 * 10**(-5))) * np.sqrt(h_freq_range / ((freq_range**2 + 79919.29) *
                                                                         (freq_range**2 + 1345600)))
    equal_loudness_contour = 20 * np.log10(np.abs(n_freq / n_freq_range))
    return equal_loudness_contour, freq_range


def generate_chirp_exp(dur, freq_start, freq_end, Fs=22050):
    """Generation chirp with exponential frequency increase

    Notebook: C1/C1S3_Dynamics.ipynb

    Args:
        dur (float): Length (seconds) of the signal
        freq_start (float): Start frequency of the chirp
        freq_end (float): End frequency of the chirp
        Fs (scalar): Sampling rate (Default value = 22050)

    Returns:
        x (np.ndarray): Generated chirp signal
        t (np.ndarray): Time axis (in seconds)
        freq (np.ndarray): Instant frequency (in Hz)
    """
    N = int(dur * Fs)
    t = np.arange(N) / Fs
    freq = np.exp(np.linspace(np.log(freq_start), np.log(freq_end), N))
    phases = np.zeros(N)
    for n in range(1, N):
        phases[n] = phases[n-1] + 2 * np.pi * freq[n-1] / Fs
    x = np.sin(phases)
    return x, t, freq


def generate_chirp_exp_equal_loudness(dur, freq_start, freq_end, Fs=22050):
    """Generation chirp with exponential frequency increase and equal loudness

    Notebook: C1/C1S3_Dynamics.ipynb

    Args:
        dur (float): Length (seconds) of the signal
        freq_start (float): Starting frequency of the chirp
        freq_end (float): End frequency of the chirp
        Fs (scalar): Sampling rate (Default value = 22050)

    Returns:
        x (np.ndarray): Generated chirp signal
        t (np.ndarray): Time axis (in seconds)
        freq (np.ndarray): Instant frequency (in Hz)
        intensity (np.ndarray): Instant intensity of the signal
    """
    N = int(dur * Fs)
    t = np.arange(N) / Fs
    intensity, freq = compute_equal_loudness_contour(freq_min=freq_start, freq_max=freq_end, num_points=N)
    amp = 10**(intensity / 20)
    phases = np.zeros(N)
    for n in range(1, N):
        phases[n] = phases[n-1] + 2 * np.pi * freq[n-1] / Fs
    x = amp * np.sin(phases)
    return x, t, freq, intensity


def compute_adsr(len_A=10, len_D=10, len_S=60, len_R=10, height_A=1.0, height_S=0.5):
    """Computation of idealized ADSR model

    Notebook: C1/C1S3_Timbre.ipynb

    Args:
        len_A (int): Length (samples) of A phase (Default value = 10)
        len_D (int): Length (samples) of D phase (Default value = 10)
        len_S (int): Length (samples) of S phase (Default value = 60)
        len_R (int): Length (samples) of R phase (Default value = 10)
        height_A (float): Height of A phase (Default value = 1.0)
        height_S (float): Height of S phase (Default value = 0.5)

    Returns:
        curve_ADSR (np.ndarray): ADSR model
    """
    curve_A = np.arange(len_A) * height_A / len_A
    curve_D = height_A - np.arange(len_D) * (height_A - height_S) / len_D
    curve_S = np.ones(len_S) * height_S
    curve_R = height_S * (1 - np.arange(1, len_R + 1) / len_R)
    curve_ADSR = np.concatenate((curve_A, curve_D, curve_S, curve_R))
    return curve_ADSR


def compute_envelope(x, win_len_sec=0.01, Fs=4000):
    """Computation of a signal's envelopes

    Notebook: C1/C1S3_Timbre.ipynb

    Args:
        x (np.ndarray): Signal (waveform) to be analyzed
        win_len_sec (float): Length (seconds) of the window (Default value = 0.01)
        Fs (scalar): Sampling rate (Default value = 4000)

    Returns:
        env (np.ndarray): Magnitude envelope
        env_upper (np.ndarray): Upper envelope
        env_lower (np.ndarray): Lower envelope
    """
    win_len_half = round(win_len_sec * Fs * 0.5)
    N = x.shape[0]
    env = np.zeros(N)
    env_upper = np.zeros(N)
    env_lower = np.zeros(N)
    for i in range(N):
        i_start = max(0, i - win_len_half)
        i_end = min(N, i + win_len_half)
        env[i] = np.amax(np.abs(x)[i_start:i_end])
        env_upper[i] = np.amax(x[i_start:i_end])
        env_lower[i] = np.amin(x[i_start:i_end])
    return env, env_upper, env_lower


def compute_plot_envelope(x, win_len_sec, Fs, figsize=(6, 3), title=''):
    """Computation and subsequent plotting of a signal's envelope

    Notebook: C1/C1S3_Timbre.ipynb

    Args:
        x (np.ndarray): Signal (waveform) to be analyzed
        win_len_sec (float): Length (seconds) of the window
        Fs (scalar): Sampling rate
        figsize (tuple): Size of the figure (Default value = (6, 3))
        title (str): Title of the figure (Default value = '')

    Returns:
        fig (mpl.figure.Figure): Generated figure
    """
    t = np.arange(x.size)/Fs
    env, env_upper, env_lower = compute_envelope(x, win_len_sec=win_len_sec, Fs=Fs)
    fig = plt.figure(figsize=figsize)
    plt.plot(t, x, color='gray', label='Waveform')
    plt.plot(t, env_upper, linewidth=2, color='cyan', label='Upper envelope')
    plt.plot(t, env_lower, linewidth=2, color='blue', label='Lower envelope')
    plt.plot(t, env, linewidth=2, color='red', label='Magnitude envelope')
    plt.title(title)
    plt.xlabel('Time (seconds)')
    plt.ylabel('Amplitude')
    plt.xlim([t[0], t[-1]])
    plt.ylim([-0.7, 0.7])
    plt.legend(loc='lower right')
    plt.show()
    ipd.display(ipd.Audio(data=x, rate=Fs))
    return fig


def generate_sinusoid_vibrato(dur=5, Fs=1000, amp=0.5, freq=440, vib_amp=1, vib_rate=5):
    """Generation of a sinusoid signal with vibrato

    Notebook: C1/C1S3_Timbre.ipynb

    Args:
        dur (float): Duration (in seconds) (Default value = 5)
        Fs (scalar): Sampling rate (Default value = 1000)
        amp (float): Amplitude of sinusoid (Default value = 0.5)
        freq (float): Frequency (Hz) of sinusoid (Default value = 440)
        vib_amp (float): Amplitude (Hz) of the frequency oscillation (Default value = 1)
        vib_rate (float): Rate (Hz) of the frequency oscillation (Default value = 5)

    Returns:
        x (np.ndarray): Generated signal
        t (np.ndarray): Time axis (in seconds)

    """
    num_samples = int(Fs * dur)
    t = np.arange(num_samples) / Fs
    freq_vib = freq + vib_amp * np.sin(t * 2 * np.pi * vib_rate)
    phase_vib = np.zeros(num_samples)
    for i in range(1, num_samples):
        phase_vib[i] = phase_vib[i-1] + 2 * np.pi * freq_vib[i-1] / Fs
    x = amp * np.sin(phase_vib)
    return x, t


def generate_sinusoid_tremolo(dur=5, Fs=1000, amp=0.5, freq=440, trem_amp=0.1, trem_rate=5):
    """Generation of a sinusoid signal with tremolo

    Notebook: C1/C1S3_Timbre.ipynb

    Args:
        dur (float): Duration (in seconds) (Default value = 5)
        Fs (scalar): Sampling rate (Default value = 1000)
        amp (float): Amplitude of sinusoid (Default value = 0.5)
        freq (float): Frequency (Hz) of sinusoid (Default value = 440)
        trem_amp (float): Amplitude of the amplitude oscillation (Default value = 0.1)
        trem_rate (float): Rate (Hz) of the amplitude oscillation (Default value = 5)

    Returns:
        x (np.ndarray): Generated signal
        t (np.ndarray): Time axis (in seconds)
    """
    num_samples = int(Fs * dur)
    t = np.arange(num_samples) / Fs
    amps = amp + trem_amp * np.sin(t * 2 * np.pi * trem_rate)
    x = amps * np.sin(2*np.pi*(freq*t))
    return x, t


def generate_tone(p=60, weight_harmonic=np.ones([16, 1]), Fs=11025, dur=2):
    """Generation of a tone with harmonics

    Notebook: C1/C1S3_Timbre.ipynb

    Args:
        p (float): MIDI pitch of the tone (Default value = 60)
        weight_harmonic (np.ndarray): Weights for the different harmonics (Default value = np.ones([16, 1])
        Fs (scalar): Sampling frequency (Default value = 11025)
        dur (float): Duration (seconds) of the signal (Default value = 2)

    Returns:
        x (np.ndarray): Generated signal
        t (np.ndarray): Time axis (in seconds)
    """
    freq = 2 ** ((p - 69) / 12) * 440
    num_samples = int(Fs * dur)
    t = np.arange(num_samples) / Fs
    x = np.zeros(t.shape)
    for h, w in enumerate(weight_harmonic):
        x = x + w * np.sin(2 * np.pi * freq * (h + 1) * t)
    return x, t


def plot_spectrogram(x, Fs=11025, N=4096, H=2048, figsize=(4, 2)):
    """Computation and subsequent plotting of the spectrogram of a signal

    Notebook: C1/C1S3_Timbre.ipynb

    Args:
        x: Signal (waveform) to be analyzed
        Fs: Sampling rate (Default value = 11025)
        N: FFT length (Default value = 4096)
        H: Hopsize (Default value = 2048)
        figsize: Size of the figure (Default value = (4, 2))

    """
    N, H = 2048, 1024
    X = librosa.stft(x, n_fft=N, hop_length=H, win_length=N, window='hann')
    Y = np.abs(X)
    plt.figure(figsize=figsize)
    librosa.display.specshow(librosa.amplitude_to_db(Y, ref=np.max),
                             y_axis='linear', x_axis='time', sr=Fs, hop_length=H, cmap='gray_r')
    plt.ylim([0, 3000])
    # plt.colorbar(format='%+2.0f dB')
    plt.xlabel('Time (seconds)')
    plt.ylabel('Frequency (Hz)')
    plt.tight_layout()
    plt.show()
