"""
Module: libfmp.c4.c4s5_evaluation
Author: Meinard Müller, Tim Zunner
License: The MIT license, https://opensource.org/licenses/MIT

This file is part of the FMP Notebooks (https://www.audiolabs-erlangen.de/FMP)
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

import libfmp.b


def measure_prf(num_TP, num_FN, num_FP):
    """Compute P, R, and F from size of TP, FN, and FP [FMP, Section 4.5.1]

    Notebook: C4/C4S5_Evaluation.ipynb

    Args:
        num_TP (int): True positives
        num_FN (int): False negative
        num_FP (int): False positives

    Returns:
        P (float): Precision
        R (float): Recall
        F (float): F-measure
    """
    P = num_TP / (num_TP + num_FP)
    R = num_TP / (num_TP + num_FN)
    if (P + R) > 0:
        F = 2 * P * R / (P + R)
    else:
        F = 0
    return P, R, F


def measure_prf_sets(I, I_ref_pos, I_est_pos, details=False):
    """Compute P, R, and F from sets I, I_ref_pos, I_est_pos [FMP, Section 4.5.1]

    Notebook: C4/C4S5_Evaluation.ipynb

    Args:
        I: Set of items
        I_ref_pos: Reference set of positive items
        I_est_pos: Set of items being estimated as positive
        details: Print details (Default value = False)

    Returns:
        P (float): Precision
        R (float): Recall
        F (float): F-measure
    """
    I_ref_neg = I.difference(I_ref_pos)
    I_est_neg = I.difference(I_est_pos)
    TP = I_est_pos.intersection(I_ref_pos)
    FN = I_est_neg.intersection(I_ref_pos)
    FP = I_est_pos.intersection(I_ref_neg)
    P, R, F = measure_prf(len(TP), len(FN), len(FP))
    if details:
        print('TP = ', TP, ';  FN = ', FN, ';  FP = ', FP)
        print('P = %0.3f;  R = %0.3f;  F = %0.3f' % (P, R, F))
    return P, R, F


def convert_ann_to_seq_label(ann):
    """Convert structure annotation with integer time positions (given in indices)
    into label sequence

    Notebook: C4/C4S5_Evaluation.ipynb

    Args:
        ann (list): Annotation (list ``[[s, t, 'label'], ...]``, with ``s``, ``t`` being integers)

    Returns:
        X (list): Sequencs of labels
    """
    X = []
    for seg in ann:
        K = seg[1] - seg[0]
        for k in range(K):
            X.append(seg[2])
    return X


def plot_seq_label(ax, X, Fs=1, color_label=[], direction='horizontal',
                   fontsize=10, time_axis=False, print_labels=True):
    """Plot label sequence in the style of annotations

    Notebook: C4/C4S5_Evaluation.ipynb

    Args:
        ax: Axis used for plotting
        X: Label sequence
        Fs: Sampling rate (Default value = 1)
        color_label: List of colors for labels (Default value = [])
        direction: Parameter used for :func:`libfmp.b.b_plot.plot_segments` (Default value = 'horizontal')
        fontsize: Parameter used for :func:`libfmp.b.b_plot.plot_segments` (Default value = 10)
        time_axis: Parameter used for :func:`libfmp.b.b_plot.plot_segments` (Default value = False)
        print_labels: Parameter used for :func:`libfmp.b.b_plot.plot_segments` (Default value = True)

    Returns:
         ann_X: Structure annotation for label sequence
    """
    ann_X = []
    for m, cur_x in enumerate(X):
        ann_X.append([(m-0.5)/Fs, (m+0.5)/Fs, cur_x])
    libfmp.b.plot_segments(ann_X, ax=ax, time_axis=time_axis, fontsize=fontsize,
                           direction=direction, colors=color_label, print_labels=print_labels)
    return ann_X


def compare_pairwise(X):
    """Compute set of positive items from label sequence [FMP, Section 4.5.3]

    Notebook: C4/C4S5_Evaluation.ipynb

    Args:
        X (list or np.ndarray): Label sequence

    Returns:
        I_pos (np.ndarray): Set of positive items
    """
    N = len(X)
    I_pos = np.zeros((N, N))
    for n in range(1, N):
        for m in range(n):
            if X[n] is X[m]:
                I_pos[n, m] = 1
    return I_pos


def evaluate_pairwise(I_ref_pos, I_est_pos):
    """Compute pairwise evaluation measures [FMP, Section 4.5.3]

    Notebook: C4/C4S5_Evaluation.ipynb

    Args:
        I_ref_pos (np.ndarray): Referenence set of positive items
        I_est_pos (np.ndarray): Set of items being estimated as positive

    Returns:
        P (float): Precision
        R (float): Recall
        F (float): F-measure
        num_TP (int): Number of true positives
        num_FN (int): Number of false negatives
        num_FP (int): Number of false positives
        I_eval (np.ndarray): Data structure encoding TP, FN, FP
    """
    I_eval = np.zeros(I_ref_pos.shape)
    TP = (I_ref_pos + I_est_pos) > 1
    FN = (I_ref_pos - I_est_pos) > 0
    FP = (I_ref_pos - I_est_pos) < 0
    I_eval[TP] = 1
    I_eval[FN] = 2
    I_eval[FP] = 3
    num_TP = np.sum(TP)
    num_FN = np.sum(FN)
    num_FP = np.sum(FP)
    P, R, F = measure_prf(num_TP, num_FN, num_FP)
    return P, R, F, num_TP, num_FN, num_FP, I_eval


def plot_matrix_label(M, X, color_label=None, figsize=(3, 3), cmap='gray_r', fontsize=8, print_labels=True):
    """Plot matrix and label sequence

    Notebook: C4/C4S5_Evaluation.ipynb

    Args:
        M: Matrix
        X: Label sequence
        color_label: List of colors for labels (Default value = None)
        figsize: Figure size (Default value = (3, 3))
        cmap: Colormap for imshow (Default value = 'gray_r')
        fontsize: Font size (Default value = 8)
        print_labels: Display labels inside Rectangles (Default value = True)

    Returns:
        fig: Handle for figure
        ax: Handle for axes
    """
    fig, ax = plt.subplots(2, 3, gridspec_kw={'width_ratios': [0.1, 1, 0.05],
                                              'wspace': 0.2, 'height_ratios': [1, 0.1]},
                           figsize=figsize)

    colorList = np.array([[1, 1, 1, 1],  [0, 0, 0, 0.7]])
    cmap = ListedColormap(colorList)
    im = ax[0, 1].imshow(M, aspect='auto', cmap=cmap,  origin='lower', interpolation='nearest')
    im.set_clim(vmin=-0.5, vmax=1.5)
    ax_cb = plt.colorbar(im, cax=ax[0, 2])
    ax_cb.set_ticks(np.arange(0, 2, 1))
    ax_cb.set_ticklabels(np.arange(0, 2, 1))
    ax[0, 1].set_xticks([])
    ax[0, 1].set_yticks([])
    plot_seq_label(ax[1, 1], X, color_label=color_label, fontsize=fontsize, print_labels=print_labels)
    ax[1, 2].axis('off')
    ax[1, 0].axis('off')
    plot_seq_label(ax[0, 0], X, color_label=color_label, fontsize=fontsize, print_labels=print_labels,
                   direction='vertical')
    return fig, ax


def plot_matrix_pairwise(I_eval, figsize=(3, 2.5)):
    """Plot matrix I_eval encoding TP, FN, FP (see :func:`libfmp.c4.c4s5_evaluation.evaluate_pairwise`)

    Notebook: C4/C4S5_Evaluation.ipynb

    Args:
        I_eval: Data structure encoding TP, FN, FP
        figsize: Figure size (Default value = (3, 2.5))

    Returns:
        fig: Handle for figure
        im: Handle for imshow
    """
    fig = plt.figure(figsize=figsize)
    colorList = np.array([[1, 1, 1, 1], [0, 0.7, 0, 1], [1, 0, 0, 1], [1, 0.5, 0.5, 1]])
    cmap = ListedColormap(colorList)
    im = plt.imshow(I_eval, aspect='auto', cmap=cmap,  origin='lower', interpolation='nearest')
    im.set_clim(vmin=-0.5, vmax=3.5)
    plt.xticks([])
    plt.yticks([])
    ax_cb = plt.colorbar(im)
    ax_cb.set_ticks(np.arange(0, 4, 1))
    ax_cb.set_ticklabels(['', 'TP', 'FN', 'FP'])
    return fig, im


def evaluate_boundary(B_ref, B_est, tau):
    """Compute boundary evaluation measures [FMP, Section 4.5.4]

    Notebook: C4/C4S5_Evaluation.ipynb

    Args:
        B_ref (np.ndarray): Reference boundary annotations
        B_est (np.ndarray): Estimated boundary annotations
        tau (int): Tolerance parameter.
            Note: Condition ``|b_{k+1}-b_k|>2tau`` should be fulfilled [FMP, Eq. 4.58]

    Returns:
        P (float): Precision
        R (float): Recall
        F (float): F-measure
        num_TP (int): Number of true positives
        num_FN (int): Number of false negatives
        num_FP (int): Number of false positives
        B_tol (np.ndarray): Data structure encoding B_ref with tolerance
        I_eval (np.ndarray): Data structure encoding TP, FN, FP
    """
    N = len(B_ref)
    num_TP = 0
    num_FN = 0
    num_FP = 0
    B_tol = np.zeros((np.array([B_ref])).shape)
    B_eval = np.zeros((np.array([B_ref])).shape)
    for n in range(N):
        min_idx = max(0, n - tau)
        max_idx = min(N - 1, n + tau)
        if B_ref[n] == 1:
            B_tol[:, min_idx:max_idx+1] = 2
            B_tol[:, n] = 1
            temp = sum(B_est[min_idx:max_idx+1])
            if temp > 0:
                num_TP += temp
            else:
                num_FN += 1
                B_eval[:, n] = 2
        if B_est[n] == 1:
            if sum(B_ref[min_idx:max_idx+1]) == 0:
                num_FP += 1
                B_eval[:, n] = 3
            else:
                B_eval[:, n] = 1
    P, R, F = measure_prf(num_TP, num_FN, num_FP)
    return P, R, F, num_TP, num_FN, num_FP, B_tol, B_eval


def plot_boundary_measures(B_ref, B_est, tau, figsize=(8, 2.5)):
    """Plot B_ref and B_est (see :func:`libfmp.c4.c4s5_evaluation.evaluate_boundary`)

    Notebook: C4/C4S5_Evaluation.ipynb

    Args:
        B_ref: Reference boundary annotations
        B_est: Estimated boundary annotations
        tau: Tolerance parameter
        figsize: Figure size (Default value = (8, 2.5))

    Returns:
        fig: Handle for figure
        ax: Handle for axes
    """
    P, R, F, num_TP, num_FN, num_FP, B_tol, B_eval = evaluate_boundary(B_ref, B_est, tau)

    colorList = np.array([[1., 1., 1., 1.], [0., 0., 0., 1.], [0.7, 0.7, 0.7, 1.]])
    cmap_tol = ListedColormap(colorList)
    colorList = np.array([[1, 1, 1, 1], [0, 0.7, 0, 1], [1, 0, 0, 1], [1, 0.5, 0.5, 1]])
    cmap_measures = ListedColormap(colorList)

    fig, ax = plt.subplots(3, 2, gridspec_kw={'width_ratios': [1, 0.02],
                                              'wspace': 0.2, 'height_ratios': [1, 1, 1]},
                           figsize=figsize)

    im = ax[0, 0].imshow(B_tol, cmap=cmap_tol, interpolation='nearest')
    ax[0, 0].set_title('Reference boundaries (with tolerance)')
    im.set_clim(vmin=-0.5, vmax=2.5)
    ax[0, 0].set_xticks([])
    ax[0, 0].set_yticks([])
    ax_cb = plt.colorbar(im, cax=ax[0, 1])
    ax_cb.set_ticks(np.arange(0, 3, 1))
    ax_cb.set_ticklabels(['', 'Positive', 'Tolerance'])

    im = ax[1, 0].imshow(np.array([B_est]), cmap=cmap_tol, interpolation='nearest')
    ax[1, 0].set_title('Estimated boundaries')
    im.set_clim(vmin=-0.5, vmax=2.5)
    ax[1, 0].set_xticks([])
    ax[1, 0].set_yticks([])
    ax_cb = plt.colorbar(im, cax=ax[1, 1])
    ax_cb.set_ticks(np.arange(0, 3, 1))
    ax_cb.set_ticklabels(['', 'Positive', 'Tolerance'])

    im = ax[2, 0].imshow(B_eval, cmap=cmap_measures, interpolation='nearest')
    ax[2, 0].set_title('Evaluation')
    im.set_clim(vmin=-0.5, vmax=3.5)
    ax[2, 0].set_xticks([])
    ax[2, 0].set_yticks([])
    ax_cb = plt.colorbar(im, cax=ax[2, 1])
    ax_cb.set_ticks(np.arange(0, 4, 1))
    ax_cb.set_ticklabels(['', 'TP', 'FN', 'FP'])
    plt.show()
    return fig, ax
