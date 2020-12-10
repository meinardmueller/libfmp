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
    Notebook: C4/C4S5_Evaluation.ipynb"""
    P = num_TP / (num_TP + num_FP)
    R = num_TP / (num_TP + num_FN)
    if (P + R) > 0:
        F = 2 * P * R / (P + R)
    else:
        F = 0
    return P, R, F


def measure_prf_sets(I, I_ref_pos, I_est_pos, details=False):
    """Compute P, R, and F from sets I, I_ref_pos, I_est_pos [FMP, Section 4.5.1]
    Notebook: C4/C4S5_Evaluation.ipynb"""
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
        ann: Annotation (list  [[s,t,'label'], ...], with s,t being integers)

    Returns:
        X: Sequencs of labels
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
        Fs: Sampling rate
        color_label: List of colors for labels
        direction, fontsize, time_axis, print_labels:
            Paramters used in libfmp.b.plot_segments

    Returns:
        ann_X: Structure annotation for label sequence
    """
    ann_X = []
    for m in range(len(X)):
        ann_X.append([(m-0.5)/Fs, (m+0.5)/Fs, X[m]])
    libfmp.b.plot_segments(ann_X, ax=ax, time_axis=time_axis, fontsize=fontsize,
                           direction=direction, colors=color_label, print_labels=print_labels)
    return ann_X


def compare_pairwise(X):
    """Compute set of positive items from label sequence [FMP, Section 4.5.3]
    Notebook: C4/C4S5_Evaluation.ipynb"""
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
        I_ref_pos, I_est_pos: Sets of positive items for reference and estimation

    Returns:
        P, R, F, num_TP, num_FN, num_FP: Evaluation measures
        I_eval: Data structure encoding TP, FN, FP
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


def plot_matrix_label(M, X, color_label=None, title='', clim=[0, 1], figsize=(3, 3),
                      cmap='gray_r', fontsize=8, print_labels=True):
    """Plot matrix and label sequence
    Notebook: C4/C4S5_Evaluation.ipynb"""
    fig, ax = plt.subplots(2, 3, gridspec_kw={'width_ratios': [0.1, 1, 0.05],
                                              'wspace': 0.2, 'height_ratios': [1, 0.1]},
                           figsize=figsize)

    colorList = np.array([[1, 1, 1, 1],  [0, 0, 0, 0.7]])
    cmap = ListedColormap(colorList)
    im = ax[0, 1].imshow(M, aspect='auto', cmap=cmap,  origin='lower')
    im.set_clim(vmin=-0.5, vmax=1.5)
    ax_cb = plt.colorbar(im, cax=ax[0, 2])
    ax_cb.set_ticks(np.arange(0, 2, 1))
    ax_cb.set_ticklabels(np.arange(0, 2, 1))
    ax[0, 1].set_xticks([])
    ax[0, 1].set_yticks([])
    plot_seq_label(ax[1, 1], X, color_label=color_label, fontsize=fontsize, print_labels=print_labels)
    ax[1, 2].axis('off'), ax[1, 0].axis('off')
    plot_seq_label(ax[0, 0], X, color_label=color_label, fontsize=fontsize, print_labels=print_labels,
                   direction='vertical')
    return fig, ax


def plot_matrix_pairwise(I_eval, figsize=(3, 2.5)):
    """Plot matrix I_eval encoding TP, FN, FP (see libfmp.c4.evaluate_pairwise)
    Notebook: C4/C4S5_Evaluation.ipynb"""
    fig = plt.figure(figsize=figsize)
    colorList = np.array([[1, 1, 1, 1], [0, 0.7, 0, 1], [1, 0, 0, 1], [1, 0.5, 0.5, 1]])
    cmap = ListedColormap(colorList)
    im = plt.imshow(I_eval, aspect='auto', cmap=cmap,  origin='lower')
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
        B_ref, B_est: Boundary annotations for reference and estimation
        tau: Tolerance parameter
             Note: Condition |b_{k+1}-b_k|>2tau should be fulfilled [FMP, Eq. 4.58]

    Returns:
        P, R, F, num_TP, num_FN, num_FP: Evaluation measures
        B_tol: Data structure encoding B_ref with tolerance
        B_eval: Data structure encoding TP, FN, FP
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


def plot_boundary_measures(B_ref, B_est, tau, figsize=(8, 2.5), details=False):
    """Plot B_tol, B_est, and B_eval (see libfmp.c4.evaluate_boundary)
    Notebook: C4/C4S5_Evaluation.ipynb"""
    P, R, F, num_TP, num_FN, num_FP, B_tol, B_eval = evaluate_boundary(B_ref, B_est, tau)

    colorList = np.array([[1., 1., 1., 1.], [0., 0., 0., 1.], [0.7, 0.7, 0.7, 1.]])
    cmap_tol = ListedColormap(colorList)
    colorList = np.array([[1, 1, 1, 1], [0, 0.7, 0, 1], [1, 0, 0, 1], [1, 0.5, 0.5, 1]])
    cmap_measures = ListedColormap(colorList)

    fig, ax = plt.subplots(3, 2, gridspec_kw={'width_ratios': [1, 0.02],
                                              'wspace': 0.2, 'height_ratios': [1, 1, 1]},
                           figsize=figsize)

    im = ax[0, 0].imshow(B_tol, cmap=cmap_tol)
    ax[0, 0].set_title('Reference boundaries (with tolerance)')
    im.set_clim(vmin=-0.5, vmax=2.5)
    ax[0, 0].set_xticks([])
    ax[0, 0].set_yticks([])
    ax_cb = plt.colorbar(im, cax=ax[0, 1])
    ax_cb.set_ticks(np.arange(0, 3, 1))
    ax_cb.set_ticklabels(['', 'Positive', 'Tolerance'])

    im = ax[1, 0].imshow(np.array([B_est]), cmap=cmap_tol)
    ax[1, 0].set_title('Estimated boundaries')
    im.set_clim(vmin=-0.5, vmax=2.5)
    ax[1, 0].set_xticks([])
    ax[1, 0].set_yticks([])
    ax_cb = plt.colorbar(im, cax=ax[1, 1])
    ax_cb.set_ticks(np.arange(0, 3, 1))
    ax_cb.set_ticklabels(['', 'Positive', 'Tolerance'])

    im = ax[2, 0].imshow(B_eval, cmap_measures)
    ax[2, 0].set_title('Evaluation')
    im.set_clim(vmin=-0.5, vmax=3.5)
    ax[2, 0].set_xticks([])
    ax[2, 0].set_yticks([])
    ax_cb = plt.colorbar(im, cax=ax[2, 1])
    ax_cb.set_ticks(np.arange(0, 4, 1))
    ax_cb.set_ticklabels(['', 'TP', 'FN', 'FP'])
    plt.show()
    return fig, ax
