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
        pitches: List of MIDI pitches
        dur: Duration (in seconds) of each sinusoid
        Fs: Sampling rate
        amp: Amplitude of generated signal

    Returns:
        x: Signal
        t: Time axis (in seconds)
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
        freq: Frequency of Shepard tone
        dur: Duration (in seconds)
        Fs: Sampling rate
        amp: Amplitude of generated signal

    Returns:
        x: Shepard tone
        t: Time axis (in seconds)
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
    x = amp * x/np.max(x)
    return x, t


def generate_chirp_exp_octave(freq_start=440, dur=8, Fs=44100, amp=1):
    """Generate one octave of a chirp with exponential frequency increase

    Notebook: C1/C1S1_ChromaShepard.ipynb

    Args:
        freq_start: Start frequency of chirp
        dur: Duration (in seconds)
        Fs: Sampling rate
        amp: Amplitude of generated signal

    Returns:
        x: chirp signal
        t: Time axis (in seconds)
    """
    N = int(dur * Fs)
    t = np.arange(N) / Fs
    x = np.sin(2 * np.pi * freq_start * np.power(2, t / dur) / np.log(2) * dur)
    x = amp * x/np.max(x)
    return x, t


def generate_shepard_glissando(num_octaves=3, dur_octave=8, Fs=44100):
    """Generate several ocatves of a Shepared glissando

    Notebook: C1/C1S1_ChromaShepard.ipynb

    Args:
        num_octaves: Number of octaves
        dur_octave: Duration (in seconds) per octave
        Fs: Sampling rate

    Returns:
        x: Shepared glissando
        t: Time axis (in seconds)
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
