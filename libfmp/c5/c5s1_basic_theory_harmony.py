"""
Module: libfmp.c5.c5s1_basic_theory_harmony
Author: Meinard Müller, Christof Weiss
License: The MIT license, https://opensource.org/licenses/MIT

This file is part of the FMP Notebooks (https://www.audiolabs-erlangen.de/FMP)
"""

import numpy as np


def generate_sinusoid_scale(pitches=[69], duration=0.5, Fs=4000, amplitude_max=0.5):
    """Generate synthetic sound of scale using sinusoids

    Notebook: C5/C5S1_Scales_CircleFifth.ipynb

    Args:
        pitches (list): List of pitchs (MIDI note numbers) (Default value = [69])
        duration (float): Duration (seconds) (Default value = 0.5)
        Fs (scalar): Sampling rate (Default value = 4000)
        amplitude_max (float): Amplitude (Default value = 0.5)

    Returns:
        x (np.ndarray): Synthesized signal
    """
    N = int(duration * Fs)
    t = np.arange(0, N) / Fs
    x = []
    for p in pitches:
        omega = 2 ** ((p - 69) / 12) * 440
        x = np.append(x, np.sin(2 * np.pi * omega * t))
    x = amplitude_max * x / np.max(x)
    return x


def generate_sinusoid_chord(pitches=[69], duration=1, Fs=4000, amplitude_max=0.5):
    """Generate synthetic sound of chord using sinusoids

    Notebook: C5/C5S1_Chords.ipynb

    Args:
        pitches (list): List of pitches (MIDI note numbers) (Default value = [69])
        duration (float): Duration (seconds) (Default value = 1)
        Fs (scalar): Sampling rate (Default value = 4000)
        amplitude_max (float): Amplitude (Default value = 0.5)

    Returns:
        x (np.ndarray): Synthesized signal
    """
    N = int(duration * Fs)
    t = np.arange(0, N) / Fs
    x = np.zeros(N)
    for p in pitches:
        omega = 2 ** ((p - 69) / 12) * 440
        x = x + np.sin(2 * np.pi * omega * t)
    x = amplitude_max * x / np.max(x)
    return x
