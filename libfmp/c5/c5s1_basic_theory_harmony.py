"""
Module: libfmp.c5.c5s1_basic_theory_harmony
Author: Meinard Müller, Christof Weiss
License: The MIT license, https://opensource.org/licenses/MIT

This file is part of the FMP Notebooks (https://www.audiolabs-erlangen.de/FMP)
"""

import numpy as np


def generate_sinusoid_scale(pitches=[69], duration=0.5, Fs=4000, amplitude_max=0.5):
    """Generate synthetic sound of scale using sinusoids

    Notebook: C5/C5S1_Chords.ipynb

    Args:
        pitches: List of pitchs (MIDI note numbers)
        duration: Duration (seconds)
        Fs: Sampling rate
        amplitude_max: Amplitude

    Returns:
        x: Synthesized signal
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
        pitches: List of pitches (MIDI note numbers)
        duration: Duration (seconds)
        Fs: Sampling rate
        amplitude_max: Amplitude

    Returns:
        x: Synthesized signal
    """
    N = int(duration * Fs)
    t = np.arange(0, N) / Fs
    x = np.zeros(N)
    for p in pitches:
        omega = 2 ** ((p - 69) / 12) * 440
        x = x + np.sin(2 * np.pi * omega * t)
    x = amplitude_max * x / np.max(x)
    return x
