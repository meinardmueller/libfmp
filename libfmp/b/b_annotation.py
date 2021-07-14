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
    """Read a CSV file in table format and creates a pd.DataFrame from it, with observations in the
    rows and variables in the columns.


    Args:
        fn (str): Filename
        header (bool): Boolean (Default value = True)
        add_label (bool): Add column with constant value of `add_label` (Default value = False)

    Returns:
        df (pd.DataFrame): Pandas DataFrame
    """
    df = pd.read_csv(fn, sep=';', keep_default_na=False, header=0 if header else None)
    if add_label:
        assert 'label' not in df.columns, 'Label column must not exist if `add_label` is True'
        df = df.assign(label=[add_label] * len(df.index))
    return df


def write_csv(df, fn, header=True):
    """Write a pd.DataFrame to a CSV file, with observations in the rows and variables in the columns.

    Args:
        df (pd.DataFrame): Pandas DataFrame
        fn (str): Filename
        header (bool): Boolean (Default value = True)
    """
    df.to_csv(fn, sep=';', index=False, quoting=2, header=header)


def cut_audio(fn_in, fn_out, start_sec, end_sec, normalize=True, write=True, Fs=22050):
    """Cut an audio file using specificed start and end time positions and writes the result to a new audio file.

    Args:
        fn_in (str): Filename and path for input audio file
        fn_out (str): Filename and path for input audio file
        start_sec (float): Start time position (in seconds) of cut
        end_sec (float): End time position (in seconds) of cut
        normalize (bool): If True, then normalize audio (with max norm) (Default value = True)
        write (bool): If True, then write audio (Default value = True)
        Fs (scalar): Sampling rate of audio (Default value = 22050)

    Returns:
        x_cut (np.ndarray): Cut audio
    """
    x_cut, Fs = librosa.load(fn_in, sr=Fs, offset=start_sec, duration=end_sec-start_sec)
    if normalize is True:
        x_cut = x_cut / np.max(np.abs(x_cut))
    if write is True:
        libfmp.b.write_audio(fn_out, x_cut, Fs)
    return x_cut


def cut_csv_file(fn_in, fn_out, start_sec, end_sec, write=True):
    """Cut a annotation CSV file (where each row corresponds to the four variables ``start``, ``end``, ``pitch``,
    and ``label``) using specificed start and end time positions and writes the result to a new CSV file.

    Args:
        fn_in (str): Filename and path for input audio file
        fn_out (str): Filename and path for input audio file
        start_sec (float): Start time position (in seconds) of cut
        end_sec (float): End time position (in seconds) of cut
        write (bool): If True, then write csv file (Default value = True)

    Returns:
        ann_cut (list): Cut annotation file
    """
    df = pd.read_csv(fn_in, sep=',', keep_default_na=False, header=None)
    ann_cut = []
    for i, (start, end, pitch, label) in df.iterrows():
        if (start > start_sec) and (start < end_sec):
            ann_cut.append([start-start_sec, min(end, end_sec)-start, int(pitch), 100, str(int(label))])

    if write:
        columns = ['Start', 'Duration', 'Pitch', 'Velocity', 'Instrument']
        df_out = pd.DataFrame(ann_cut, columns=columns)
        df_out['Start'] = df_out['Start'].map('{:,.3f}'.format)
        df_out['Duration'] = df_out['Duration'].map('{:,.3f}'.format)
        df_out.to_csv(fn_out, sep=';', index=False)
    return ann_cut
