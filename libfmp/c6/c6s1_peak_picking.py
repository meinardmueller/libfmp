"""
Module: libfmp.c6.c6s1_peak_picking
Author: Angel Villar Corrales, Meinard Mueller
License: The MIT license, https://opensource.org/licenses/MIT

This file is part of the FMP Notebooks (https://www.audiolabs-erlangen.de/FMP)
"""

import numpy as np
from scipy.ndimage import filters


def peak_picking_boeck(activations, threshold=0.5, fps=100, include_scores=False, combine=0,
                       pre_avg=12, post_avg=6, pre_max=6, post_max=6):
    """Detects peaks.

    Implements the peak-picking method described in:
    "Evaluating the Online Capabilities of Onset Detection Methods"
    Sebastian Boeck, Florian Krebs and Markus Schedl
    Proceedings of the 13th International Society for Music Information Retrieval Conference (ISMIR), 2012

    Modified by Jan Schlueter, 2014-04-24

    Parameters
    ----------
    activations : np.nadarray
        vector of activations to process
    threshold : float
        threshold for peak-picking
    fps : float
        frame rate of onset activation function in Hz
    include_scores : boolean
        include activation for each returned peak
    combine :
        only report 1 onset for N seconds
    pre_avg :
        use N past seconds for moving average
    post_avg :
        use N future seconds for moving average
    pre_max :
        use N past seconds for moving maximum
    post_max :
        use N future seconds for moving maximum

    Returns
    -------
    stamps : np.ndarray
    """

    import scipy.ndimage.filters as sf
    activations = activations.ravel()

    # detections are activations equal to the moving maximum
    max_length = int((pre_max + post_max) * fps) + 1
    if max_length > 1:
        max_origin = int((pre_max - post_max) * fps / 2)
        mov_max = sf.maximum_filter1d(activations, max_length, mode='constant', origin=max_origin)
        detections = activations * (activations == mov_max)
    else:
        detections = activations

    # detections must be greater than or equal to the moving average + threshold
    avg_length = int((pre_avg + post_avg) * fps) + 1
    if avg_length > 1:
        avg_origin = int((pre_avg - post_avg) * fps / 2)
        mov_avg = sf.uniform_filter1d(activations, avg_length, mode='constant', origin=avg_origin)
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
    """Computes the positive peaks of the input vector x
       Python adaption from the Matlab Roeder_Peak_Picking script peaks.m from the Chroma Toolbox
       reckjn 2017

    Parameters
    ----------

    x                 signal to be searched for (positive) peaks

    dir               +1 for forward peak searching, -1 for backward peak
                      searching. default is dir == -1.

    abs_thresh        absolute threshold signal, i.e. only peaks
                      satisfying x(i)>=abs_thresh(i) will be reported.
                      abs_thresh must have the same number of samples as x.
                      a sensible choice for this parameter would be a global or local
                      average or median of the signal x.
                      if omitted, half the median of x will be used.

    rel_thresh        relative threshold signal. only peak positions i with an
                      uninterrupted positive ascent before position i of at least
                      rel_thresh(i) and a possibly interrupted (see parameter descent_thresh)
                      descent of at least rel_thresh(i) will be reported.
                      rel_thresh must have the same number of samples as x.
                      a sensible choice would be some measure related to the
                      global or local variance of the signal x.
                      if omitted, half the standard deviation of x will be used.

    descent_thresh    descent threshold. during peak candidate verfication, if a slope change
                      from negative to positive slope occurs at sample i BEFORE the descent has
                      exceeded rel_thresh(i), and if descent_thresh(i) has not been exceeded yet,
                      the current peak candidate will be dropped.
                      this situation corresponds to a secondary peak
                      occuring shortly after the current candidate peak (which might lead
                      to a higher peak value)!

                      the value descent_thresh(i) must not be larger than rel_thresh(i).

                      descent_thresh must have the same number of samples as x.
                      a sensible choice would be some measure related to the
                      global or local variance of the signal x.
                      if omitted, 0.5*rel_thresh will be used.

    tmin              index of start sample. peak search will begin at x(tmin).

    tmax              index of end sample. peak search will end at x(tmax).


    Returns
    -------

    peaks               array of peak positions

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

        if (dy >= 0):
            rise = rise + dy
        else:
            descent = descent + dy

        if (dyold >= 0):
            if (dy < 0):  # slope change positive->negative
                if (rise >= rel_thresh[cur_idx] and searching_peak is True):
                    candidate = cur_idx
                    searching_peak = False
                riseold = rise
                rise = 0
        else:  # dyold < 0
            if (dy < 0):  # in descent
                if (descent <= -rel_thresh[candidate] and searching_peak is False):
                    if (x[candidate] >= abs_thresh[candidate]):
                        P.append(candidate)  # verified candidate as True peak
                    searching_peak = True
            else:  # dy >= 0 slope change negative->positive
                if searching_peak is False:  # currently verifying a peak
                    if (x[candidate] - x[cur_idx] <= descent_thresh[cur_idx]):
                        rise = riseold + descent  # skip intermediary peak
                    if (descent <= -rel_thresh[candidate]):
                        if x[candidate] >= abs_thresh[candidate]:
                            P.append(candidate)    # verified candidate as True peak
                    searching_peak = True
                descent = 0
        dyold = dy
    peaks = np.array(P)
    return peaks


# msaf implementation of peak picking for foote
def peak_picking_nieto(x, median_len=16, offset_rel=0.05, sigma=4):
    """Obtain peaks from a novelty curve using an adaptive threshold."""
    offset = x.mean() * offset_rel
    x = filters.gaussian_filter1d(x, sigma=sigma)
    threshold_local = filters.median_filter(x, size=median_len) + offset
    peaks = []
    for i in range(1, x.shape[0] - 1):
        if x[i - 1] < x[i] and x[i] > x[i + 1]:
            if x[i] > threshold_local[i]:
                peaks.append(i)
    peaks = np.array(peaks)
    return peaks
