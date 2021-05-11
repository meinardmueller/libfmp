"""
Module: libfmp.c6.c6s1_peak_picking
Author: Angel Villar Corrales, Meinard Mueller
License: The MIT license, https://opensource.org/licenses/MIT

This file is part of the FMP Notebooks (https://www.audiolabs-erlangen.de/FMP)
"""

import numpy as np
from scipy.ndimage import filters


def peak_picking_simple(x, threshold=None):
    """Peak picking strategy looking for positions with increase followed by descrease

    Notebook: C6/C6S1_PeakPicking.ipynb

    Args:
        x (np.ndarray): Input function
        threshold (float): Lower threshold for peak to survive

    Returns:
        peaks (np.ndarray): Array containing peak positions
    """
    peaks = []
    if threshold is None:
        threshold = np.min(x) - 1
    for i in range(1, x.shape[0] - 1):
        if x[i - 1] < x[i] and x[i] > x[i + 1]:
            if x[i] >= threshold:
                peaks.append(i)
    peaks = np.array(peaks)
    return peaks


def peak_picking_boeck(activations, threshold=0.5, fps=100, include_scores=False, combine=False,
                       pre_avg=12, post_avg=6, pre_max=6, post_max=6):
    """Detects peaks.

    | Implements the peak-picking method described in:
    | "Evaluating the Online Capabilities of Onset Detection Methods"
    | Sebastian Boeck, Florian Krebs and Markus Schedl
    | Proceedings of the 13th International Society for Music Information Retrieval Conference (ISMIR), 2012

    Modified by Jan Schlueter, 2014-04-24

    Args:
        activations (np.nadarray): Vector of activations to process
        threshold (float): Threshold for peak-picking (Default value = 0.5)
        fps (scalar): Frame rate of onset activation function in Hz (Default value = 100)
        include_scores (bool): Include activation for each returned peak (Default value = False)
        combine (bool): Only report 1 onset for N seconds (Default value = False)
        pre_avg (float): Use N past seconds for moving average (Default value = 12)
        post_avg (float): Use N future seconds for moving average (Default value = 6)
        pre_max (float): Use N past seconds for moving maximum (Default value = 6)
        post_max (float): Use N future seconds for moving maximum (Default value = 6)

    Returns:
        peaks (np.ndarray): Peak positions
    """

    activations = activations.ravel()

    # detections are activations equal to the moving maximum
    max_length = int((pre_max + post_max) * fps) + 1
    if max_length > 1:
        max_origin = int((pre_max - post_max) * fps / 2)
        mov_max = filters.maximum_filter1d(activations, max_length, mode='constant', origin=max_origin)
        detections = activations * (activations == mov_max)
    else:
        detections = activations

    # detections must be greater than or equal to the moving average + threshold
    avg_length = int((pre_avg + post_avg) * fps) + 1
    if avg_length > 1:
        avg_origin = int((pre_avg - post_avg) * fps / 2)
        mov_avg = filters.uniform_filter1d(activations, avg_length, mode='constant', origin=avg_origin)
        detections = detections * (detections >= mov_avg + threshold)
    else:
        # if there is no moving average, treat the threshold as a global one
        detections = detections * (detections >= threshold)

    # convert detected onsets to a list of timestamps
    if combine:
        stamps = []
        last_onset = 0
        for i in np.nonzero(detections)[0]:
            # only report an onset if the last N frames none was reported
            if i > last_onset + combine:
                stamps.append(i)
                # save last reported onset
                last_onset = i
        stamps = np.array(stamps)
    else:
        stamps = np.where(detections)[0]

    # include corresponding activations per peak if needed
    if include_scores:
        scores = activations[stamps]
        if avg_length > 1:
            scores -= mov_avg[stamps]
        return stamps / float(fps), scores
    else:
        return stamps / float(fps)


def peak_picking_roeder(x, direction=None, abs_thresh=None, rel_thresh=None, descent_thresh=None, tmin=None, tmax=None):
    """| Computes the positive peaks of the input vector x
    | Python adaption from the Matlab Roeder_Peak_Picking script peaks.m from the internal Sync Toolbox
    | reckjn 2017

    Args:
        x (np.nadarray): Signal to be searched for (positive) peaks
        direction (int): +1 for forward peak searching, -1 for backward peak searching.
            default is dir == -1. (Default value = None)
        abs_thresh (float): Absolute threshold signal, i.e. only peaks
            satisfying x(i)>=abs_thresh(i) will be reported.
            abs_thresh must have the same number of samples as x.
            a sensible choice for this parameter would be a global or local
            average or median of the signal x.
            If omitted, half the median of x will be used. (Default value = None)
        rel_thresh (float): Relative threshold signal. Only peak positions i with an
            uninterrupted positive ascent before position i of at least
            rel_thresh(i) and a possibly interrupted (see parameter descent_thresh)
            descent of at least rel_thresh(i) will be reported.
            rel_thresh must have the same number of samples as x.
            A sensible choice would be some measure related to the
            global or local variance of the signal x.
            if omitted, half the standard deviation of W will be used.
        descent_thresh (float): Descent threshold. during peak candidate verfication, if a slope change
            from negative to positive slope occurs at sample i BEFORE the descent has
            exceeded rel_thresh(i), and if descent_thresh(i) has not been exceeded yet,
            the current peak candidate will be dropped.
            this situation corresponds to a secondary peak
            occuring shortly after the current candidate peak (which might lead
            to a higher peak value)!
            |
            | The value descent_thresh(i) must not be larger than rel_thresh(i).
            |
            | descent_thresh must have the same number of samples as x.
            a sensible choice would be some measure related to the
            global or local variance of the signal x.
            if omitted, 0.5*rel_thresh will be used. (Default value = None)
        tmin (int): Index of start sample. peak search will begin at x(tmin). (Default value = None)
        tmax (int): Index of end sample. peak search will end at x(tmax). (Default value = None)

    Returns:
        peaks (np.nadarray): Array of peak positions
    """

    # set default values
    if direction is None:
        direction = -1
    if abs_thresh is None:
        abs_thresh = np.tile(0.5*np.median(x), len(x))
    if rel_thresh is None:
        rel_thresh = 0.5*np.tile(np.sqrt(np.var(x)), len(x))
    if descent_thresh is None:
        descent_thresh = 0.5*rel_thresh
    if tmin is None:
        tmin = 1
    if tmax is None:
        tmax = len(x)

    dyold = 0
    dy = 0
    rise = 0  # current amount of ascent during a rising portion of the signal x
    riseold = 0  # accumulated amount of ascent from the last rising portion of x
    descent = 0  # current amount of descent (<0) during a falling portion of the signal x
    searching_peak = True
    candidate = 1
    P = []

    if direction == 1:
        my_range = np.arange(tmin, tmax)
    elif direction == -1:
        my_range = np.arange(tmin, tmax)
        my_range = my_range[::-1]

    # run through x
    for cur_idx in my_range:
        # get local gradient
        dy = x[cur_idx+direction] - x[cur_idx]

        if dy >= 0:
            rise = rise + dy
        else:
            descent = descent + dy

        if dyold >= 0:
            if dy < 0:  # slope change positive->negative
                if rise >= rel_thresh[cur_idx] and searching_peak is True:
                    candidate = cur_idx
                    searching_peak = False
                riseold = rise
                rise = 0
        else:  # dyold < 0
            if dy < 0:  # in descent
                if descent <= -rel_thresh[candidate] and searching_peak is False:
                    if x[candidate] >= abs_thresh[candidate]:
                        P.append(candidate)  # verified candidate as True peak
                    searching_peak = True
            else:  # dy >= 0 slope change negative->positive
                if searching_peak is False:  # currently verifying a peak
                    if x[candidate] - x[cur_idx] <= descent_thresh[cur_idx]:
                        rise = riseold + descent  # skip intermediary peak
                    if descent <= -rel_thresh[candidate]:
                        if x[candidate] >= abs_thresh[candidate]:
                            P.append(candidate)    # verified candidate as True peak
                    searching_peak = True
                descent = 0
        dyold = dy
    peaks = np.array(P)
    return peaks


def peak_picking_MSAF(x, median_len=16, offset_rel=0.05, sigma=4.0):
    """Peak picking strategy following MSFA using an adaptive threshold (https://github.com/urinieto/msaf)

    Notebook: C6/C6S1_PeakPicking.ipynb

    Args:
        x (np.ndarray): Input function
        median_len (int): Length of media filter used for adaptive thresholding (Default value = 16)
        offset_rel (float): Additional offset used for adaptive thresholding (Default value = 0.05)
        sigma (float): Variance for Gaussian kernel used for smoothing the novelty function (Default value = 4.0)

    Returns:
        peaks (np.ndarray): Peak positions
        x (np.ndarray): Local threshold
        threshold_local (np.ndarray): Filtered novelty curve
    """
    offset = x.mean() * offset_rel
    x = filters.gaussian_filter1d(x, sigma=sigma)
    threshold_local = filters.median_filter(x, size=median_len) + offset
    peaks = []
    for i in range(1, x.shape[0] - 1):
        if x[i - 1] < x[i] and x[i] > x[i + 1]:
            if x[i] > threshold_local[i]:
                peaks.append(i)
    peaks = np.array(peaks)
    return peaks, x, threshold_local
