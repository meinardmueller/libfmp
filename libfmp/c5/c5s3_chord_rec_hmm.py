"""
Module: libfmp.c5.c5s3_chord_rec_hmm
Author: Meinard Müller, Christof Weiss
License: The MIT license, https://opensource.org/licenses/MIT

This file is part of the FMP Notebooks (https://www.audiolabs-erlangen.de/FMP)
"""

import numpy as np
from sklearn.preprocessing import normalize
from scipy.linalg import circulant
from numba import jit
from matplotlib import pyplot as plt

import libfmp.c3
from libfmp.c5 import get_chord_labels


def generate_sequence_hmm(N, A, C, B, details=False):
    """Generate observation and state sequence from given HMM

    Notebook: C5/C5S3_HiddenMarkovModel.ipynb

    Args:
        N (int): Number of observations to be generated
        A (np.ndarray): State transition probability matrix of dimension I x I
        C (np.ndarray): Initial state distribution of dimension I
        B (np.ndarray): Output probability matrix of dimension I x K
        details (bool): If "True" then shows details (Default value = False)

    Returns:
        O (np.ndarray): Observation sequence of length N
        S (np.ndarray): State sequence of length N
    """
    assert N > 0, "N should be at least one"
    I = A.shape[1]
    K = B.shape[1]
    assert I == A.shape[0], "A should be an I-square matrix"
    assert I == C.shape[0], "Dimension of C should be I"
    assert I == B.shape[0], "Column-dimension of B should be I"

    O = np.zeros(N, int)
    S = np.zeros(N, int)
    for n in range(N):
        if n == 0:
            i = np.random.choice(np.arange(I), p=C)
        else:
            i = np.random.choice(np.arange(I), p=A[i, :])
        k = np.random.choice(np.arange(K), p=B[i, :])
        S[n] = i
        O[n] = k
        if details:
            print('n = %d, S[%d] = %d, O[%d] = %d' % (n, n, S[n], n, O[n]))
    return O, S


def estimate_hmm_from_o_s(O, S, I, K):
    """Estimate the state transition and output probability matrices from
    a given observation and state sequence

    Notebook: C5/C5S3_HiddenMarkovModel.ipynb

    Args:
        O (np.ndarray): Observation sequence of length N
        S (np.ndarray): State sequence of length N
        I (int): Number of states
        K (int): Number of observation symbols

    Returns:
        A_est (np.ndarray): State transition probability matrix of dimension I x I
        B_est (np.ndarray): Output probability matrix of dimension I x K
    """
    # Estimate A
    A_est = np.zeros([I, I])
    N = len(S)
    for n in range(N-1):
        i = S[n]
        j = S[n+1]
        A_est[i, j] += 1
    A_est = normalize(A_est, axis=1, norm='l1')

    # Estimate B
    B_est = np.zeros([I, K])
    for i in range(I):
        for k in range(K):
            B_est[i, k] = np.sum(np.logical_and(S == i, O == k))
    B_est = normalize(B_est, axis=1, norm='l1')
    return A_est, B_est


@jit(nopython=True)
def viterbi(A, C, B, O):
    """Viterbi algorithm for solving the uncovering problem

    Notebook: C5/C5S3_Viterbi.ipynb

    Args:
        A (np.ndarray): State transition probability matrix of dimension I x I
        C (np.ndarray): Initial state distribution  of dimension I
        B (np.ndarray): Output probability matrix of dimension I x K
        O (np.ndarray): Observation sequence of length N

    Returns:
        S_opt (np.ndarray): Optimal state sequence of length N
        D (np.ndarray): Accumulated probability matrix
        E (np.ndarray): Backtracking matrix
    """
    I = A.shape[0]    # Number of states
    N = len(O)  # Length of observation sequence

    # Initialize D and E matrices
    D = np.zeros((I, N))
    E = np.zeros((I, N-1)).astype(np.int32)
    D[:, 0] = np.multiply(C, B[:, O[0]])

    # Compute D and E in a nested loop
    for n in range(1, N):
        for i in range(I):
            temp_product = np.multiply(A[:, i], D[:, n-1])
            D[i, n] = np.max(temp_product) * B[i, O[n]]
            E[i, n-1] = np.argmax(temp_product)

    # Backtracking
    S_opt = np.zeros(N).astype(np.int32)
    S_opt[-1] = np.argmax(D[:, -1])
    for n in range(N-2, -1, -1):
        S_opt[n] = E[int(S_opt[n+1]), n]

    return S_opt, D, E


@jit(nopython=True)
def viterbi_log(A, C, B, O):
    """Viterbi algorithm (log variant) for solving the uncovering problem

    Notebook: C5/C5S3_Viterbi.ipynb

    Args:
        A (np.ndarray): State transition probability matrix of dimension I x I
        C (np.ndarray): Initial state distribution  of dimension I
        B (np.ndarray): Output probability matrix of dimension I x K
        O (np.ndarray): Observation sequence of length N

    Returns:
        S_opt (np.ndarray): Optimal state sequence of length N
        D_log (np.ndarray): Accumulated log probability matrix
        E (np.ndarray): Backtracking matrix
    """
    I = A.shape[0]    # Number of states
    N = len(O)  # Length of observation sequence
    tiny = np.finfo(0.).tiny
    A_log = np.log(A + tiny)
    C_log = np.log(C + tiny)
    B_log = np.log(B + tiny)

    # Initialize D and E matrices
    D_log = np.zeros((I, N))
    E = np.zeros((I, N-1)).astype(np.int32)
    D_log[:, 0] = C_log + B_log[:, O[0]]

    # Compute D and E in a nested loop
    for n in range(1, N):
        for i in range(I):
            temp_sum = A_log[:, i] + D_log[:, n-1]
            D_log[i, n] = np.max(temp_sum) + B_log[i, O[n]]
            E[i, n-1] = np.argmax(temp_sum)

    # Backtracking
    S_opt = np.zeros(N).astype(np.int32)
    S_opt[-1] = np.argmax(D_log[:, -1])
    for n in range(N-2, -1, -1):
        S_opt[n] = E[int(S_opt[n+1]), n]

    return S_opt, D_log, E


def plot_transition_matrix(A, log=True, ax=None, figsize=(6, 5), title='',
                           xlabel='State (chord label)', ylabel='State (chord label)',
                           cmap='gray_r', quadrant=False):
    """Plot a transition matrix for 24 chord models (12 major and 12 minor triads)

    Notebook: C5/C5S3_ChordRec_HMM.ipynb

    Args:
        A: Transition matrix
        log: Show log probabilities (Default value = True)
        ax: Axis (Default value = None)
        figsize: Width, height in inches (only used when ax=None) (Default value = (6, 5))
        title: Title for plot (Default value = '')
        xlabel: Label for x-axis (Default value = 'State (chord label)')
        ylabel: Label for y-axis (Default value = 'State (chord label)')
        cmap: Color map (Default value = 'gray_r')
        quadrant: Plots additional lines for C-major and C-minor quadrants (Default value = False)

    Returns:
        fig: The created matplotlib figure or None if ax was given.
        ax: The used axes.
        im: The image plot
    """
    fig = None
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=figsize)
        ax = [ax]

    if log is True:
        A_plot = np.log(A)
        cbar_label = 'Log probability'
        clim = [-6, 0]
    else:
        A_plot = A
        cbar_label = 'Probability'
        clim = [0, 1]
    im = ax[0].imshow(A_plot, origin='lower', aspect='equal', cmap=cmap, interpolation='nearest')
    im.set_clim(clim)
    plt.sca(ax[0])
    cbar = plt.colorbar(im)
    ax[0].set_xlabel(xlabel)
    ax[0].set_ylabel(ylabel)
    ax[0].set_title(title)
    cbar.ax.set_ylabel(cbar_label)

    chord_labels = get_chord_labels()
    chord_labels_squeezed = chord_labels.copy()
    for k in [1, 3, 6, 8, 10, 11, 13, 15, 17, 18, 20, 22]:
        chord_labels_squeezed[k] = ''

    ax[0].set_xticks(np.arange(24))
    ax[0].set_yticks(np.arange(24))
    ax[0].set_xticklabels(chord_labels_squeezed)
    ax[0].set_yticklabels(chord_labels)

    if quadrant is True:
        ax[0].axvline(x=11.5, ymin=0, ymax=24, linewidth=2, color='r')
        ax[0].axhline(y=11.5, xmin=0, xmax=24, linewidth=2, color='r')

    return fig, ax, im


def matrix_circular_mean(A):
    """Computes circulant matrix with mean diagonal sums

    Notebook: C5/C5S3_ChordRec_HMM.ipynb

    Args:
        A (np.ndarray): Square matrix

    Returns:
        A_mean (np.ndarray): Circulant output matrix
    """
    N = A.shape[0]
    A_shear = np.zeros((N, N))
    for n in range(N):
        A_shear[:, n] = np.roll(A[:, n], -n)
    circ_sum = np.sum(A_shear, axis=1)
    A_mean = circulant(circ_sum) / N
    return A_mean


def matrix_chord24_trans_inv(A):
    """Computes transposition-invariant matrix for transition matrix
    based 12 major chords and 12 minor chords

    Notebook: C5/C5S3_ChordRec_HMM.ipynb

    Args:
        A (np.ndarray): Input transition matrix

    Returns:
        A_ti (np.ndarray): Output transition matrix
    """
    A_ti = np.zeros(A.shape)
    A_ti[0:12, 0:12] = matrix_circular_mean(A[0:12, 0:12])
    A_ti[0:12, 12:24] = matrix_circular_mean(A[0:12, 12:24])
    A_ti[12:24, 0:12] = matrix_circular_mean(A[12:24, 0:12])
    A_ti[12:24, 12:24] = matrix_circular_mean(A[12:24, 12:24])
    return A_ti


def uniform_transition_matrix(p=0.01, N=24):
    """Computes uniform transition matrix

    Notebook: C5/C5S3_ChordRec_HMM.ipynb

    Args:
        p (float): Self transition probability (Default value = 0.01)
        N (int): Column and row dimension (Default value = 24)

    Returns:
        A (np.ndarray): Output transition matrix
    """
    off_diag_entries = (1-p) / (N-1)     # rows should sum up to 1
    A = off_diag_entries * np.ones([N, N])
    np.fill_diagonal(A, p)
    return A


@jit(nopython=True)
def viterbi_log_likelihood(A, C, B_O):
    """Viterbi algorithm (log variant) for solving the uncovering problem

    Notebook: C5/C5S3_Viterbi.ipynb

    Args:
        A (np.ndarray): State transition probability matrix of dimension I x I
        C (np.ndarray): Initial state distribution  of dimension I
        B_O (np.ndarray): Likelihood matrix of dimension I x N

    Returns:
        S_opt (np.ndarray): Optimal state sequence of length N
        S_mat (np.ndarray): Binary matrix representation of optimal state sequence
        D_log (np.ndarray): Accumulated log probability matrix
        E (np.ndarray): Backtracking matrix
    """
    I = A.shape[0]    # Number of states
    N = B_O.shape[1]  # Length of observation sequence
    tiny = np.finfo(0.).tiny
    A_log = np.log(A + tiny)
    C_log = np.log(C + tiny)
    B_O_log = np.log(B_O + tiny)

    # Initialize D and E matrices
    D_log = np.zeros((I, N))
    E = np.zeros((I, N-1)).astype(np.int32)
    D_log[:, 0] = C_log + B_O_log[:, 0]

    # Compute D and E in a nested loop
    for n in range(1, N):
        for i in range(I):
            temp_sum = A_log[:, i] + D_log[:, n-1]
            D_log[i, n] = np.max(temp_sum) + B_O_log[i, n]
            E[i, n-1] = np.argmax(temp_sum)

    # Backtracking
    S_opt = np.zeros(N).astype(np.int32)
    S_opt[-1] = np.argmax(D_log[:, -1])
    for n in range(N-2, -1, -1):
        S_opt[n] = E[int(S_opt[n+1]), n]

    # Matrix representation of result
    S_mat = np.zeros((I, N)).astype(np.int32)
    for n in range(N):
        S_mat[S_opt[n], n] = 1

    return S_mat, S_opt, D_log, E


def chord_recognition_all(X, ann_matrix, p=0.15, filt_len=None, filt_type='mean'):
    """Conduct template- and HMM-based chord recognition and evaluates the approaches

    Notebook: C5/C5S3_ChordRec_Beatles.ipynb

    Args:
        X (np.ndarray): Chromagram
        ann_matrix (np.ndarray): Reference annotation as given as time-chord binary matrix
        p (float): Self-transition probability used for HMM (Default value = 0.15)
        filt_len (int): Filter length used for prefilitering (Default value = None)
        filt_type (str): Filter type used for prefilitering (Default value = 'mean')

    Returns:
        result_Tem (tuple): Chord recogntion evaluation results ([P, R, F, TP, FP, FN]) for template-based approach
        result_HMM (tuple): Chord recogntion evaluation results ([P, R, F, TP, FP, FN]) for HMM-based approach
        chord_Tem (np.ndarray): Template-based chord recogntion result given as binary matrix
        chord_HMM (np.ndarray): HMM-based chord recogntion result given as binary matrix
        chord_sim (np.ndarray): Chord similarity matrix
    """
    if filt_len is not None:
        if filt_type == 'mean':
            X, Fs_X = libfmp.c3.smooth_downsample_feature_sequence(X, Fs=1, filt_len=filt_len, down_sampling=1)
        if filt_type == 'median':
            X, Fs_X = libfmp.c3.median_downsample_feature_sequence(X, Fs=1, filt_len=filt_len, down_sampling=1)
    # Template-based chord recogntion
    chord_sim, chord_Tem = libfmp.c5.chord_recognition_template(X, norm_sim='1')
    result_Tem = libfmp.c5.compute_eval_measures(ann_matrix, chord_Tem)
    # HMM-based chord recogntion
    A = libfmp.c5.uniform_transition_matrix(p=p)
    C = 1 / 24 * np.ones((1, 24))
    B_O = chord_sim
    chord_HMM, _, _, _ = libfmp.c5.viterbi_log_likelihood(A, C, B_O)
    result_HMM = libfmp.c5.compute_eval_measures(ann_matrix, chord_HMM)
    return result_Tem, result_HMM, chord_Tem, chord_HMM, chord_sim
