"""
Module: libfmp.b.b_audio
Author: Frank Zalkow, Meinard Mueller
License: The MIT license, https://opensource.org/licenses/MIT

This file is part of the FMP Notebooks (https://www.audiolabs-erlangen.de/FMP)
"""

import librosa
import soundfile as sf
import IPython.display as ipd
import pandas as pd


def read_audio(path, Fs=None, mono=False):
    """Read an audio file into a np.ndarray.

    Args:
        path (str): Path to audio file
        Fs (scalar): Resample audio to given sampling rate. Use native sampling rate if None. (Default value = None)
        mono (bool): Convert multi-channel file to mono. (Default value = False)

    Returns:
        x (np.ndarray): Waveform signal
        Fs (scalar): Sampling rate
    """
    return librosa.load(path, sr=Fs, mono=mono)


def write_audio(path, x, Fs):
    """Write a signal (as np.ndarray) to an audio file.

    Args:
        path (str): Path to audio file
        x (np.ndarray): Waveform signal
        Fs (scalar): Sampling rate
    """
    sf.write(path, x, Fs)


def audio_player_list(signals, rates, width=270, height=40, columns=None, column_align='center'):
    """Generate a list of HTML audio players tags for a given list of audio signals.

    Notebook: B/B_PythonAudio.ipynb

    Args:
        signals (list): List of audio signals
        rates (list): List of sample rates
        width (int): Width of player (either number or list) (Default value = 270)
        height (int): Height of player (either number or list) (Default value = 40)
        columns (list): Column headings (Default value = None)
        column_align (str): Left, center, right (Default value = 'center')
    """
    pd.set_option('display.max_colwidth', None)

    if isinstance(width, int):
        width = [width] * len(signals)
    if isinstance(height, int):
        height = [height] * len(signals)

    audio_list = []
    for cur_x, cur_Fs, cur_width, cur_height in zip(signals, rates, width, height):
        audio_html = ipd.Audio(data=cur_x, rate=cur_Fs)._repr_html_()
        audio_html = audio_html.replace('\n', '').strip()
        audio_html = audio_html.replace('<audio ', f'<audio style="width: {cur_width}px; height: {cur_height}px" ')
        audio_list.append([audio_html])

    df = pd.DataFrame(audio_list, index=columns).T
    table_html = df.to_html(escape=False, index=False, header=bool(columns))
    table_html = table_html.replace('<th>', f'<th style="text-align: {column_align}">')
    ipd.display(ipd.HTML(table_html))
