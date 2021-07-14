"""
Module: libfmp.c4.c4s1_annotation
Author: Meinard Müller
License: The MIT license, https://opensource.org/licenses/MIT

This file is part of the FMP Notebooks (https://www.audiolabs-erlangen.de/FMP)
"""
import numpy as np

import libfmp.b


def get_color_for_annotation_file(filename):
    """Gets color dict for annotation file. This function is specialized for some specfic files used in the FMP
    notebooks, i.e.:

    * FMP_C4_Audio_Brahms_HungarianDances-05_Ormandy
    * FMP_C6_Audio_Brahms_HungarianDances-05_Ormandy
    * FMP_C4_F13_ZagerEvans_InTheYear2525
    * FMP_C6_Audio_ZagerEvans_InTheYear2525

    Args:
        filename (str): Annotation file

    Returns:
        color_ann (dict): Dictionary encoding color scheme
    """
    color_ann = None
    if filename == 'FMP_C4_Audio_Brahms_HungarianDances-05_Ormandy.csv':
        color_ann = {'A1': [1, 0, 0, 0.2], 'A2': [1, 0, 0, 0.2], 'A3': [1, 0, 0, 0.2],
                     'B1': [0, 1, 0, 0.2], 'B2': [0, 1, 0, 0.2], 'B3': [0, 1, 0, 0.2],
                     'B4': [0, 1, 0, 0.2], 'C': [0, 0, 1, 0.2], '': [1, 1, 1, 0]}
    if filename == 'FMP_C6_Audio_Brahms_HungarianDances-05_Ormandy.csv':
        color_ann = {'A1': [1, 0, 0, 0.2], 'A2': [1, 0, 0, 0.2], 'A3': [1, 0, 0, 0.2],
                     'B1': [0, 1, 0, 0.2], 'B2': [0, 1, 0, 0.2], 'B3': [0, 1, 0, 0.2],
                     'B4': [0, 1, 0, 0.2], 'C': [0, 0, 1, 0.2], '': [1, 1, 1, 0]}
    if filename == 'FMP_C4_F13_ZagerEvans_InTheYear2525.csv':
        color_ann = {'I': [0, 1, 0, 0.2], 'V1': [1, 0, 0, 0.2], 'V2': [1, 0, 0, 0.2],
                     'V3': [1, 0, 0, 0.2], 'V4': [1, 0, 0, 0.2], 'V5': [1, 0, 0, 0.2],
                     'V6': [1, 0, 0, 0.2], 'V7': [1, 0, 0, 0.2], 'V8': [1, 0, 0, 0.2],
                     'B': [0, 0, 1, 0.2], 'O': [1, 1, 0, 0.2], '': [1, 1, 1, 0]}
    if filename == 'FMP_C6_Audio_ZagerEvans_InTheYear2525.csv':
        color_ann = {'I': [0, 1, 0, 0.2], 'V1': [1, 0, 0, 0.2], 'V2': [1, 0, 0, 0.2],
                     'V3': [1, 0, 0, 0.2], 'V4': [1, 0, 0, 0.2], 'V5': [1, 0, 0, 0.2],
                     'V6': [1, 0, 0, 0.2], 'V7': [1, 0, 0, 0.2], 'V8': [1, 0, 0, 0.2],
                     'B': [0, 0, 1, 0.2], 'O': [1, 1, 0, 0.2], '': [1, 1, 1, 0]}
    return color_ann


def convert_structure_annotation(ann, Fs=1, remove_digits=False, index=False):
    """Convert structure annotations

    Notebook: C4/C4S1_MusicStructureGeneral.ipynb

    Args:
        ann (list): Structure annotions
        Fs (scalar): Sampling rate (Default value = 1)
        remove_digits (bool): Remove digits from labels (Default value = False)
        index (bool): Round to nearest integer (Default value = False)

    Returns:
        ann_converted (list): Converted annotation
    """
    ann_converted = []
    for r in ann:
        s = r[0] * Fs
        t = r[1] * Fs
        if index:
            s = int(np.round(s))
            t = int(np.round(t))
        if remove_digits:
            label = ''.join([i for i in r[2] if not i.isdigit()])
        else:
            label = r[2]
        ann_converted = ann_converted + [[s, t, label]]
    return ann_converted


def read_structure_annotation(fn_ann, fn_ann_color='', Fs=1, remove_digits=False, index=False):
    """Read and convert structure annotation and colors

    Notebook: C4/C4S1_MusicStructureGeneral.ipynb

    Args:
        fn_ann (str): Path and filename for structure annotions
        fn_ann_color (str): Filename used to identify colors (Default value = '')
        Fs (scalar): Sampling rate (Default value = 1)
        remove_digits (bool): Remove digits from labels (Default value = False)
        index (bool): Round to nearest integer (Default value = False)

    Returns:
        ann (list): Annotations
        color_ann (dict): Color scheme
    """
    df = libfmp.b.read_csv(fn_ann)
    ann = [(start, end, label) for i, (start, end, label) in df.iterrows()]
    ann = convert_structure_annotation(ann, Fs=Fs, remove_digits=remove_digits, index=index)
    color_ann = {}
    if len(fn_ann_color) > 0:
        color_ann = get_color_for_annotation_file(fn_ann_color)
        if remove_digits:
            color_ann_reduced = {}
            for key, value in color_ann.items():
                key_new = ''.join([i for i in key if not i.isdigit()])
                color_ann_reduced[key_new] = value
            color_ann = color_ann_reduced
    return ann, color_ann
