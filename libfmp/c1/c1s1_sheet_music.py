"""
Module: libfmp.c1.c1s1_sheet_music
Author: Meinard Müller
License: The MIT license, https://opensource.org/licenses/MIT

This file is part of the FMP Notebooks (https://www.audiolabs-erlangen.de/FMP)
"""

import numpy as np


def generate_sinusoid_pitches(pitches=[69], dur=0.5, Fs=4000, amp=1):
    """Generation of sinusoids for a given list of MIDI pitches

    Notebook: C1/C1S1_MusicalNotesPitches.ipynb

    Args:
        pitches (list): List of MIDI pitches (Default value = [69])
        dur (float): Duration (in seconds) of each sinusoid (Default value = 0.5)
        Fs (scalar): Sampling rate (Default value = 4000)
        amp (float): Amplitude of generated signal (Default value = 1)

    Returns:
        x (np.ndarray): Signal
        t (np.ndarray): Time axis (in seconds)
    """
    N = int(dur * Fs)
    t = np.arange(N) / Fs
    x = []
    for p in pitches:
        freq = 2 ** ((p - 69) / 12) * 440
        x = np.append(x, np.sin(2 * np.pi * freq * t))
    x = amp * x / np.max(x)
    return x, t


def generate_shepard_tone(freq=440, dur=0.5, Fs=44100, amp=1):
    """Generate Shepard tone

    Notebook: C1/C1S1_ChromaShepard.ipynb

    Args:
        freq (float): Frequency of Shepard tone (Default value = 440)
        dur (float): Duration (in seconds) (Default value = 0.5)
        Fs (scalar): Sampling rate (Default value = 44100)
        amp (float): Amplitude of generated signal (Default value = 1)

    Returns:
        x (np.ndarray): Shepard tone
        t (np.ndarray): Time axis (in seconds)
    """
    N = int(dur * Fs)
    t = np.arange(N) / Fs
    num_sin = 1
    x = np.sin(2 * np.pi * freq * t)
    freq_lower = freq / 2
    while freq_lower > 20:
        num_sin += 1
        x = x + np.sin(2 * np.pi * freq_lower * t)
        freq_lower = freq_lower / 2
    freq_upper = freq * 2
    while freq_upper < 20000:
        num_sin += 1
        x = x + np.sin(2 * np.pi * freq_upper * t)
        freq_upper = freq_upper * 2
    x = x / num_sin
    x = amp * x / np.max(x)
    return x, t


def generate_chirp_exp_octave(freq_start=440, dur=8, Fs=44100, amp=1):
    """Generate one octave of a chirp with exponential frequency increase

    Notebook: C1/C1S1_ChromaShepard.ipynb

    Args:
        freq_start (float): Start frequency of chirp (Default value = 440)
        dur (float): Duration (in seconds) (Default value = 8)
        Fs (scalar): Sampling rate (Default value = 44100)
        amp (float): Amplitude of generated signal (Default value = 1)

    Returns:
        x (np.ndarray): Chirp signal
        t (np.ndarray): Time axis (in seconds)
    """
    N = int(dur * Fs)
    t = np.arange(N) / Fs
    x = np.sin(2 * np.pi * freq_start * np.power(2, t / dur) / np.log(2) * dur)
    x = amp * x / np.max(x)
    return x, t


def generate_shepard_glissando(num_octaves=3, dur_octave=8, Fs=44100):
    """Generate several ocatves of a Shepared glissando

    Notebook: C1/C1S1_ChromaShepard.ipynb

    Args:
        num_octaves (int): Number of octaves (Default value = 3)
        dur_octave (int): Duration (in seconds) per octave (Default value = 8)
        Fs (scalar): Sampling rate (Default value = 44100)

    Returns:
        x (np.ndarray): Shepared glissando
        t (np.ndarray): Time axis (in seconds)
    """
    freqs_start = 10 * 2**np.arange(0, 11)
    # Generate Shepard glissando by superimposing chirps that differ by octaves
    for freq in freqs_start:
        if freq == 10:
            x, t = generate_chirp_exp_octave(freq_start=freq, dur=dur_octave, Fs=Fs, amp=1)
        else:
            chirp, t = generate_chirp_exp_octave(freq_start=freq, dur=dur_octave, Fs=Fs, amp=1)
            x = x + chirp
    x = x / len(freqs_start)
    # Concatenate several octaves
    x = np.tile(x, num_octaves)
    N = len(x)
    t = np.arange(N) / Fs
    return x, t
