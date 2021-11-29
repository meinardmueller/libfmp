"""
Module: libfmp.c8.c8s3_nmf
Author: Meinard Müller, Tim Zunner
License: The MIT license, https://opensource.org/licenses/MIT

This file is part of the FMP Notebooks (https://www.audiolabs-erlangen.de/FMP)
"""

import numpy as np
import matplotlib.pyplot as plt
from numba import jit


@jit(nopython=True)
def nmf(V, R, thresh=0.001, L=1000, W=None, H=None, norm=False, report=False):
    """NMF algorithm with Euclidean distance

    Notebook: C8/C8S3_NMFbasic.ipynb

    Args:
        V (np.ndarray): Nonnegative matrix of size K x N
        R (int): Rank parameter
        thresh (float): Threshold used as stop criterion (Default value = 0.001)
        L (int): Maximal number of iteration (Default value = 1000)
        W (np.ndarray): Nonnegative matrix of size K x R used for initialization (Default value = None)
        H (np.ndarray): Nonnegative matrix of size R x N used for initialization (Default value = None)
        norm (bool): Applies max-normalization of columns of final W (Default value = False)
        report (bool): Reports errors during runtime (Default value = False)

    Returns:
        W (np.ndarray): Nonnegative matrix of size K x R
        H (np.ndarray): Nonnegative matrix of size R x N
        V_approx (np.ndarray): Nonnegative matrix W*H of size K x N
        V_approx_err (float): Error between V and V_approx
        H_W_error (np.ndarray): History of errors of subsequent H and W matrices
    """
    K = V.shape[0]
    N = V.shape[1]
    if W is None:
        W = np.random.rand(K, R)
    if H is None:
        H = np.random.rand(R, N)
    V = V.astype(np.float64)
    W = W.astype(np.float64)
    H = H.astype(np.float64)
    H_W_error = np.zeros((2, L))
    ell = 1
    below_thresh = False
    eps_machine = np.finfo(np.float32).eps
    while not below_thresh and ell <= L:
        H_ell = H
        W_ell = W
        H = H * (W.transpose().dot(V) / (W.transpose().dot(W).dot(H) + eps_machine))
        W = W * (V.dot(H.transpose()) / (W.dot(H).dot(H.transpose()) + eps_machine))

        # H+1 = H *p ((W^T * V) / p (W^T * W * H))
        # H = np.multiply(H, np.divide(np.matmul(np.transpose(W), V),
        #                              np.matmul(np.matmul(np.transpose(W), W), H)))
        # W+1 = W *p ((V * H^T) / p (W * H * H^T))
        # W = np.multiply(W, np.divide(np.matmul(V, np.transpose(H)),
        #                              np.matmul(np.matmul(W, H), np.transpose(H))))

        H_error = np.linalg.norm(H-H_ell, ord=2)
        W_error = np.linalg.norm(W - W_ell, ord=2)
        H_W_error[:, ell-1] = [H_error, W_error]
        if report:
            print('Iteration: ', ell, ', H_error: ', H_error, ', W_error: ', W_error)
        if H_error < thresh and W_error < thresh:
            below_thresh = True
            H_W_error = H_W_error[:, 0:ell]
        ell += 1
    if norm:
        for r in range(R):
            v_max = np.max(W[:, r])
            if v_max > 0:
                W[:, r] = W[:, r] / v_max
                H[r, :] = H[r, :] * v_max
    V_approx = W.dot(H)
    V_approx_err = np.linalg.norm(V-V_approx, ord=2)
    return W, H, V_approx, V_approx_err, H_W_error


def plot_nmf_factors(W, H, V, Fs, N_fft, H_fft, freq_max, label_pitch=None,
                     title_W='W', title_H='H', title_V='V', figsize=(13, 3)):
    """Plots the factore of an NMF-based spectral decomposition

    Notebook: C8/C8S3_NMFSpecFac.ipynb

    Args:
        W: Template matrix
        H: Activation matrix
        V: Reconstructed input matrix
        Fs: Sampling frequency
        N_fft: FFT length
        H_fft: Hopsize
        freq_max: Maximum frequency to be plotted
        label_pitch: Labels for the different pitches (Default value = None)
        title_W: Title for imshow of matrix W (Default value = 'W')
        title_H: Title for imshow of matrix H (Default value = 'H')
        title_V: Title for imshow of matrix V (Default value = 'V')
        figsize: Size of the figure (Default value = (13, 3))
    """
    R = W.shape[1]
    N = H.shape[1]
    # cmap = libfmp.b.compressed_gray_cmap(alpha=5)
    cmap = 'gray_r'
    dur_sec = (N-1) * H_fft / Fs
    if label_pitch is None:
        label_pitch = np.arange(R)

    plt.figure(figsize=figsize)
    plt.subplot(131)
    plt.imshow(W, cmap=cmap, origin='lower', aspect='auto', extent=[0, R, 0, Fs/2], interpolation='nearest')
    plt.ylim([0, freq_max])
    plt.colorbar()
    plt.xticks(np.arange(R) + 0.5, label_pitch)
    plt.xlabel('Pitch')
    plt.ylabel('Frequency (Hz)')
    plt.title(title_W)

    plt.subplot(132)
    plt.imshow(H, cmap=cmap, origin='lower', aspect='auto', extent=[0, dur_sec, 0, R], interpolation='nearest')
    plt.colorbar()
    plt.yticks(np.arange(R) + 0.5, label_pitch)
    plt.xlabel('Time (seconds)')
    plt.ylabel('Pitch')
    plt.title(title_H)

    plt.subplot(133)
    plt.imshow(V, cmap=cmap, origin='lower', aspect='auto', extent=[0, dur_sec, 0, Fs/2], interpolation='nearest')
    plt.ylim([0, freq_max])
    plt.colorbar()
    plt.xlabel('Time (seconds)')
    plt.ylabel('Frequency (Hz)')
    plt.title(title_V)

    plt.tight_layout()
    plt.show()


def pitch_from_annotation(annotation):
    """Extract set of occurring pitches from annotation

    Notebook: C8/C8S3_NMFSpecFac.ipynb

    Args:
        annotation (list): Annotation data

    Returns:
        pitch_set (np.ndarray): Set of occurring pitches
    """
    pitch_all = np.array([c[2] for c in annotation])
    pitch_set = np.unique(pitch_all)
    return pitch_set


def template_pitch(K, pitch, freq_res, tol_pitch=0.05):
    """Defines spectral template for a given pitch

    Notebook: C8/C8S3_NMFSpecFac.ipynb

    Args:
        K (int): Number of frequency points
        pitch (float): Fundamental pitch
        freq_res (float): Frequency resolution
        tol_pitch (float): Relative frequency tolerance for the harmonics (Default value = 0.05)

    Returns:
        template (np.ndarray): Nonnegative template vector of size K
    """
    max_freq = K * freq_res
    pitch_freq = 2**((pitch - 69) / 12) * 440
    max_order = int(np.ceil(max_freq / ((1 - tol_pitch) * pitch_freq)))
    template = np.zeros(K)
    for m in range(1, max_order + 1):
        min_idx = max(0, int((1 - tol_pitch) * m * pitch_freq / freq_res))
        max_idx = min(K-1, int((1 + tol_pitch) * m * pitch_freq / freq_res))
        template[min_idx:max_idx+1] = 1 / m
    return template


def init_nmf_template_pitch(K, pitch_set, freq_res, tol_pitch=0.05):
    """Initializes template matrix for a given set of pitches

    Notebook: C8/C8S3_NMFSpecFac.ipynb

    Args:
        K (int): Number of frequency points
        pitch_set (np.ndarray): Set of fundamental pitches
        freq_res (float): Frequency resolution
        tol_pitch (float): Relative frequency tolerance for the harmonics (Default value = 0.05)

    Returns:
        W (np.ndarray): Nonnegative matrix of size K x R with R = len(pitch_set)
    """
    R = len(pitch_set)
    W = np.zeros((K, R))
    for r in range(R):
        W[:, r] = template_pitch(K, pitch_set[r], freq_res, tol_pitch=tol_pitch)
    return W


def init_nmf_activation_score(N, annotation, frame_res, tol_note=[0.2, 0.5], pitch_set=None):
    """Initializes activation matrix for given score annotations

    Notebook: C8/C8S3_NMFSpecFac.ipynb

    Args:
        N (int): Number of frames
        annotation (list): Annotation data
        frame_res (time): Time resolution
        tol_note (list or np.ndarray): Tolerance (seconds) for beginning and end of a note (Default value = [0.2, 0.5])
        pitch_set (np.ndarray): Set of occurring pitches (Default value = None)

    Returns:
        H (np.ndarray): Nonnegative matrix of size R x N
        pitch_set (np.ndarray): Set of occurring pitches
    """
    note_start = np.array([c[0] for c in annotation])
    note_dur = np.array([c[1] for c in annotation])
    pitch_all = np.array([c[2] for c in annotation])
    if pitch_set is None:
        pitch_set = np.unique(pitch_all)
    R = len(pitch_set)
    H = np.zeros((R, N))
    for i in range(len(note_start)):
        start_idx = max(0, int((note_start[i] - tol_note[0]) / frame_res))
        end_idx = min(N, int((note_start[i] + note_dur[i] + tol_note[1]) / frame_res))
        pitch_idx = np.argwhere(pitch_set == pitch_all[i])
        H[pitch_idx, start_idx:end_idx] = 1
    return H, pitch_set


def init_nmf_template_pitch_onset(K, pitch_set, freq_res, tol_pitch=0.05):
    """Initializes template matrix with onsets for a given set of pitches

    Notebook: C8/C8S3_NMFSpecFac.ipynb

    Args:
        K (int): Number of frequency points
        pitch_set (np.ndarray): Set of fundamental pitches
        freq_res (float): Frequency resolution
        tol_pitch (float): Relative frequency tolerance for the harmonics (Default value = 0.05)

    Returns:
        W (np.ndarray): Nonnegative matrix of size K x (2R) with R = len(pitch_set)
    """
    R = len(pitch_set)
    W = np.zeros((K, 2*R))
    for r in range(R):
        W[:, 2*r] = 0.1
        W[:, 2*r+1] = template_pitch(K, pitch_set[r], freq_res, tol_pitch=tol_pitch)
    return W


def init_nmf_activation_score_onset(N, annotation, frame_res, tol_note=[0.2, 0.5], tol_onset=[0.3, 0.1],
                                    pitch_set=None):
    """Initializes activation matrix with onsets for given score annotations

    Notebook: C8/C8S3_NMFSpecFac.ipynb

    Args:
        N (int): Number of frames
        annotation (list): Annotation data
        frame_res (float): Time resolution
        tol_note (list or np.ndarray): Tolerance (seconds) for beginning and end of a note (Default value = [0.2, 0.5])
        tol_onset (list or np.ndarray): Tolerance (seconds) for beginning and end of an onset
            (Default value = [0.3, 0.1])
        pitch_set (np.ndarray): Set of occurring pitches (Default value = None)

    Returns:
        H (np.ndarray): Nonnegative matrix of size (2R) x N
        pitch_set (np.ndarray): Set of occurring pitches
        label_pitch (np.ndarray): Pitch labels for the templates
    """
    note_start = np.array([c[0] for c in annotation])
    note_dur = np.array([c[1] for c in annotation])
    pitch_all = np.array([c[2] for c in annotation])
    if pitch_set is None:
        pitch_set = np.unique(pitch_all)
    R = len(pitch_set)
    H = np.zeros((2*R, N))
    for i in range(len(note_start)):
        start_idx = max(0, int((note_start[i] - tol_note[0]) / frame_res))
        end_idx = min(N, int((note_start[i] + note_dur[i] + tol_note[1]) / frame_res))
        start_onset_idx = max(0, int((note_start[i] - tol_onset[0]) / frame_res))
        end_onset_idx = min(N, int((note_start[i] + tol_onset[1]) / frame_res))
        pitch_idx = np.argwhere(pitch_set == pitch_all[i])
        H[2*pitch_idx, start_onset_idx:end_onset_idx] = 1
        H[2*pitch_idx+1, start_idx:end_idx] = 1
    label_pitch = np.zeros(2*len(pitch_set),  dtype=int)
    for k in range(len(pitch_set)):
        label_pitch[2*k] = pitch_set[k]
        label_pitch[2*k+1] = pitch_set[k]
    return H, pitch_set, label_pitch


def split_annotation_lh_rh(ann):
    """Splitting of the annotation data in left and right hand

    Notebook: C8/C8S3_NMFAudioDecomp.ipynb

    Args:
        ann (list): Annotation data

    Returns:
        ann_lh (list): Annotation data for left hand
        ann_rh (list): Annotation data for right hand
    """
    ann_lh = []
    ann_rh = []
    for a in ann:
        if a[4] == 'lh':
            ann_lh = ann_lh + [a]
        if a[4] == 'rh':
            ann_rh = ann_rh + [a]
    return ann_lh, ann_rh
