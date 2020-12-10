"""
Module: libfmp.b.b_annotation
Author: Frank Zalkow, Meinard Mueller
License: The MIT license, https://opensource.org/licenses/MIT

This file is part of the FMP Notebooks (https://www.audiolabs-erlangen.de/FMP)
"""

import numpy as np
import pandas as pd
import librosa

import libfmp.b


def read_csv(fn, header=True, add_label=False):
    """Reads a CSV file

    Args:
        fn: Filename
        header: Boolean
        add_label: Add column with constant value of `add_label`

    Returns:
        df: Pandas DataFrame
    """
    df = pd.read_csv(fn, sep=';', keep_default_na=False, header=0 if header else None)
    if add_label:
        assert 'label' not in df.columns, 'Label column must not exist if `add_label` is True'
        df = df.assign(label=[add_label] * len(df.index))
    return df


def write_csv(df, fn, header=True):
    """Writes a CSV file

    Args:
        df: Pandas DataFrame
        fn: Filename
        header: Boolean
    """
    df.to_csv(fn, sep=';', index=False, quoting=2, header=header)


def cut_audio(fn_in, fn_out, start_sec, end_sec, normalize=True, write=True, Fs=22050):
    """Cuts an audio file

    Notebook: B/B_Annotations_cut.ipynb

    Args:
        fn_in: Filename and path for input audio file
        fn_out: Filename and path for input audio file
        start_sec: Start time position (in seconds) of cut
        end_sec: End time position (in seconds) of cut
        normalize: If True, then normalize audio (with max norm)
        write: If True, then write audio
        Fs: Sampling rate of audio

    Returns:
        x_cut: Cut audio
    """
    x_cut, Fs = librosa.load(fn_in, sr=Fs, offset=start_sec, duration=end_sec-start_sec)
    if normalize is True:
        x_cut = x_cut / np.max(np.abs(x_cut))
    if write is True:
        libfmp.b.write_audio(fn_out, x_cut, Fs)
    return x_cut


def cut_csv_file(fn_in, fn_out, start_sec, end_sec, write=True):
    """Cuts csv annotation file

    Notebook: B/B_Annotations_cut.ipynb

    Args:
        fn_in: Filename and path for input audio file
        fn_out: Filename and path for input audio file
        start_sec: Start time position (in seconds) of cut
        end_sec: End time position (in seconds) of cut
        write: If True, then write audio

    Returns:
        ann_cut: Cut annotation file
    """
    df = pd.read_csv(fn_in, sep=',', keep_default_na=False, header=None)
    ann_cut = []
    for i, (start, end, pitch, label) in df.iterrows():
        if (start > start_sec) and (start < end_sec):
            ann_cut.append([start-start_sec, min(end, end_sec)-start, int(pitch), 100, str(int(label))])
    columns = ['Start', 'Duration', 'Pitch', 'Velocity', 'Instrument']
    df_out = pd.DataFrame(ann_cut, columns=columns)
    df_out['Start'] = df_out['Start'].map('{:,.3f}'.format)
    df_out['Duration'] = df_out['Duration'].map('{:,.3f}'.format)
    df_out.to_csv(fn_out, sep=';', index=False)
    return ann_cut
