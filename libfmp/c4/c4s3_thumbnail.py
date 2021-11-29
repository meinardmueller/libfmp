"""
Module: libfmp.c4.c4s3_thumbnail
Author: Meinard Müller, Angel Villar-Corrales
License: The MIT license, https://opensource.org/licenses/MIT

This file is part of the FMP Notebooks (https://www.audiolabs-erlangen.de/FMP)
"""
import math
import numpy as np
from numba import jit
import matplotlib
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap

import libfmp.b
import libfmp.c4


def colormap_penalty(penalty=-2, cmap=libfmp.b.compressed_gray_cmap(alpha=5)):
    """Extend colormap with white color between the penalty value and zero

    Notebook: C4/C4S3_AudioThumbnailing.ipynb

    Args:
        penalty (float): Negative number (Default value = -2.0)
        cmap (mpl.colors.Colormap): Original colormap (Default value = libfmp.b.compressed_gray_cmap(alpha=5))

    Returns:
        cmap_penalty (mpl.colors.Colormap): Extended colormap
    """
    if isinstance(cmap, str):
        cmap = matplotlib.cm.get_cmap(cmap, 128)
    cmap_matrix = cmap(np.linspace(0, 1, 128))[:, :3]
    num_row = int(np.abs(penalty)*128)
    # cmap_penalty = np.flip(np.concatenate((cmap_matrix, np.ones((num_row, 3))), axis=0), axis=0)
    cmap_penalty = np.concatenate((np.ones((num_row, 3)), cmap_matrix), axis=0)
    cmap_penalty = ListedColormap(cmap_penalty)

    return cmap_penalty


def normalization_properties_ssm(S):
    """Normalizes self-similartiy matrix to fulfill S(n,n)=1.
    Yields a warning if max(S)<=1 is not fulfilled

    Notebook: C4/C4S3_AudioThumbnailing.ipynb

    Args:
        S (np.ndarray): Self-similarity matrix (SSM)

    Returns:
        S_normalized (np.ndarray): Normalized self-similarity matrix
    """
    S_normalized = S.copy()
    N = S_normalized.shape[0]
    for n in range(N):
        S_normalized[n, n] = 1
        max_S = np.max(S_normalized)
    if max_S > 1:
        print('Normalization condition for SSM not fulfill (max > 1)')
    return S_normalized


def plot_ssm_ann(S, ann, Fs=1, cmap='gray_r', color_ann=[], ann_x=True, ann_y=True,
                 fontsize=12, figsize=(5, 4.5), xlabel='', ylabel='', title=''):
    """Plot SSM and annotations (horizontal and vertical as overlay)

    Notebook: C4/C4S3_AudioThumbnailing.ipynb

    Args:
        S: Self-similarity matrix
        ann: Annotations
        Fs: Feature rate of path_family (Default value = 1)
        cmap: Color map for S (Default value = 'gray_r')
        color_ann: color scheme used for annotations (see :func:`libfmp.b.b_plot.plot_segments`)
            (Default value = [])
        ann_x: Plot annotations on x-axis (Default value = True)
        ann_y: Plot annotations on y-axis (Default value = True)
        fontsize: Font size used for annotation labels (Default value = 12)
        figsize: Size of figure (Default value = (5, 4.5))
        xlabel: Label for x-axis (Default value = '')
        ylabel: Label for y-axis (Default value = '')
        title: Figure size (Default value = '')

    Returns:
        fig: Handle for figure
        ax: Handle for axes
        im: Handle for imshow
    """
    fig, ax = plt.subplots(2, 2, gridspec_kw={'width_ratios': [1, 0.05],
                                              'height_ratios': [1, 0.1]}, figsize=figsize)

    fig_im, ax_im, im = libfmp.b.plot_matrix(S, Fs=Fs, Fs_F=Fs,
                                             ax=[ax[0, 0], ax[0, 1]], cmap=cmap,
                                             xlabel='', ylabel='', title='')
    ax[0, 0].set_ylabel(ylabel)
    ax[0, 0].set_xlabel(xlabel)
    ax[0, 0].set_title(title)
    if ann_y:
        libfmp.b.plot_segments_overlay(ann, ax=ax_im[0], direction='vertical',
                                       time_max=S.shape[0]/Fs, print_labels=False,
                                       colors=color_ann, alpha=0.05)
    if ann_x:
        libfmp.b.plot_segments(ann, ax=ax[1, 0], time_max=S.shape[0]/Fs, colors=color_ann,
                               time_axis=False, fontsize=fontsize)
    else:
        ax[1, 0].axis('off')
    ax[1, 1].axis('off')
    plt.tight_layout()
    return fig, ax, im


def plot_path_family(ax, path_family, Fs=1, x_offset=0, y_offset=0, proj_x=True, w_x=7, proj_y=True, w_y=7):
    """Plot path family into a given axis

    Notebook: C4/C4S3_AudioThumbnailing.ipynb

    Args:
        ax: Axis of plot
        path_family: Path family
        Fs: Feature rate of path_family (Default value = 1)
        x_offset: Offset x-axis (Default value = 0)
        y_offset: Yffset x-axis (Default value = 0)
        proj_x: Display projection on x-axis (Default value = True)
        w_x: Width used for projection on x-axis (Default value = 7)
        proj_y: Display projection on y-axis (Default value = True)
        w_y: Width used for projection on y-axis (Default value = 7)
    """
    for path in path_family:
        y = [(path[i][0] + y_offset)/Fs for i in range(len(path))]
        x = [(path[i][1] + x_offset)/Fs for i in range(len(path))]
        ax.plot(x, y, "o", color=[0, 0, 0], linewidth=3, markersize=5)
        ax.plot(x, y, '.', color=[0.7, 1, 1], linewidth=2, markersize=6)
    if proj_y:
        for path in path_family:
            y1 = path[0][0]/Fs
            y2 = path[-1][0]/Fs
            ax.add_patch(plt.Rectangle((0, y1), w_y, y2-y1, linewidth=1,
                                       facecolor=[0, 1, 0], edgecolor=[0, 0, 0]))
            # ax.plot([0, 0], [y1, y2], linewidth=8, color=[0, 1, 0])
    if proj_x:
        for path in path_family:
            x1 = (path[0][1] + x_offset)/Fs
            x2 = (path[-1][1] + x_offset)/Fs
            ax.add_patch(plt.Rectangle((x1, 0), x2-x1, w_x, linewidth=1,
                                       facecolor=[0, 0, 1], edgecolor=[0, 0, 0]))
            # ax.plot([x1, x2], [0, 0], linewidth=8, color=[0, 0, 1])


def compute_induced_segment_family_coverage(path_family):
    """Compute induced segment family and coverage from path family

    Notebook: C4/C4S3_AudioThumbnailing.ipynb

    Args:
        path_family (list): Path family

    Returns:
        segment_family (np.ndarray): Induced segment family
        coverage (float): Coverage of path family
    """
    num_path = len(path_family)
    coverage = 0
    if num_path > 0:
        segment_family = np.zeros((num_path, 2), dtype=int)
        for n in range(num_path):
            segment_family[n, 0] = path_family[n][0][0]
            segment_family[n, 1] = path_family[n][-1][0]
            coverage = coverage + segment_family[n, 1] - segment_family[n, 0] + 1
    else:
        segment_family = np.empty

    return segment_family, coverage


@jit(nopython=True)
def compute_accumulated_score_matrix(S_seg):
    """Compute the accumulated score matrix

    Notebook: C4/C4S3_AudioThumbnailing.ipynb

    Args:
        S_seg (np.ndarray): Submatrix of an enhanced and normalized SSM ``S``.
            Note: ``S`` must satisfy ``S(n,m) <= 1 and S(n,n) = 1``

    Returns:
        D (np.ndarray): Accumulated score matrix
        score (float): Score of optimal path family
    """
    inf = math.inf
    N = S_seg.shape[0]
    M = S_seg.shape[1]+1

    # Iinitializing score matrix
    D = -inf * np.ones((N, M), dtype=np.float64)
    D[0, 0] = 0.
    D[0, 1] = D[0, 0] + S_seg[0, 0]

    # Dynamic programming
    for n in range(1, N):
        D[n, 0] = max(D[n-1, 0], D[n-1, -1])
        D[n, 1] = D[n, 0] + S_seg[n, 0]
        for m in range(2, M):
            D[n, m] = S_seg[n, m-1] + max(D[n-1, m-1], D[n-1, m-2], D[n-2, m-1])

    # Score of optimal path family
    score = np.maximum(D[N-1, 0], D[N-1, M-1])

    return D, score


@jit(nopython=True)
def compute_optimal_path_family(D):
    """Compute an optimal path family given an accumulated score matrix

    Notebook: C4/C4S3_AudioThumbnailing.ipynb

    Args:
        D (np.ndarray): Accumulated score matrix

    Returns:
        path_family (list): Optimal path family consisting of list of paths
            (each path being a list of index pairs)
    """
    # Initialization
    inf = math.inf
    N = int(D.shape[0])
    M = int(D.shape[1])

    path_family = []
    path = []

    n = N - 1
    if(D[n, M-1] < D[n, 0]):
        m = 0
    else:
        m = M-1
        path_point = (N-1, M-2)
        path.append(path_point)

    # Backtracking
    while n > 0 or m > 0:

        # obtaining the set of possible predecesors given our current position
        if(n <= 2 and m <= 2):
            predecessors = [(n-1, m-1)]
        elif(n <= 2 and m > 2):
            predecessors = [(n-1, m-1), (n-1, m-2)]
        elif(n > 2 and m <= 2):
            predecessors = [(n-1, m-1), (n-2, m-1)]
        else:
            predecessors = [(n-1, m-1), (n-2, m-1), (n-1, m-2)]

        # case for the first row. Only horizontal movements allowed
        if n == 0:
            cell = (0, m-1)
        # case for the elevator column: we can keep going down the column or jumping to the end of the next row
        elif m == 0:
            if D[n-1, M-1] > D[n-1, 0]:
                cell = (n-1, M-1)
                path_point = (n-1, M-2)
                if(len(path) > 0):
                    path.reverse()
                    path_family.append(path)
                path = [path_point]
            else:
                cell = (n-1, 0)
        # case for m=1, only horizontal steps to the elevator column are allowed
        elif m == 1:
            cell = (n, 0)
        # regular case
        else:

            # obtaining the best of the possible predecesors
            max_val = -inf
            for i, cur_predecessor in enumerate(predecessors):
                if(max_val < D[cur_predecessor[0], cur_predecessor[1]]):
                    max_val = D[cur_predecessor[0], cur_predecessor[1]]
                    cell = cur_predecessor

            # saving the point in the current path
            path_point = (cell[0], cell[1]-1)
            path.append(path_point)

        (n, m) = cell

    # adding last path to the path family
    path.reverse()
    path_family.append(path)
    path_family.reverse()

    return path_family


def compute_fitness(path_family, score, N):
    """Compute fitness measure and other metrics from path family

    Notebook: C4/C4S3_AudioThumbnailing.ipynb

    Args:
        path_family (list): Path family
        score (float): Score
        N (int): Length of feature sequence

    Returns:
        fitness (float): Fitness
        score (float): Score
        score_n (float): Normalized score
        coverage (float): Coverage
        coverage_n (float): Normalized coverage
        path_family_length (int): Length of path family (total number of cells)
    """
    eps = 1e-16
    num_path = len(path_family)
    M = path_family[0][-1][1] + 1

    # Normalized score
    path_family_length = 0
    for n in range(num_path):
        path_family_length = path_family_length + len(path_family[n])
    score_n = (score - M) / (path_family_length + eps)

    # Normalized coverage
    segment_family, coverage = compute_induced_segment_family_coverage(path_family)
    coverage_n = (coverage - M) / (N + eps)

    # Fitness measure
    fitness = 2 * score_n * coverage_n / (score_n + coverage_n + eps)

    return fitness, score, score_n, coverage, coverage_n, path_family_length


def plot_ssm_ann_optimal_path_family(S, ann, seg, Fs=1, cmap='gray_r', color_ann=[], fontsize=12,
                                     figsize=(5, 4.5), ylabel=''):
    """Plot SSM, annotations, and computed optimal path family

    Notebook: C4/C4S3_AudioThumbnailing.ipynb

    Args:
        S: Self-similarity matrix
        ann: Annotations
        seg: Segment for computing the optimal path family
        Fs: Feature rate of path_family (Default value = 1)
        cmap: Color map for S (Default value = 'gray_r')
        color_ann: color scheme used for annotations (see :func:`libfmp.b.b_plot.plot_segments`)
            (Default value = [])
        fontsize: Font size used for annotation labels (Default value = 12)
        figsize: Size of figure (Default value = (5, 4.5))
        ylabel: Label for y-axis (Default value = '')

    Returns:
        fig: Handle for figure
        ax: Handle for axes
        im: Handle for imshow
    """
    N = S.shape[0]
    S_seg = S[:, seg[0]:seg[1]+1]
    D, score = compute_accumulated_score_matrix(S_seg)
    path_family = compute_optimal_path_family(D)
    fitness, score, score_n, coverage, coverage_n, path_family_length = compute_fitness(
        path_family, score, N)
    title = r'$\bar{\sigma}(\alpha)=%0.2f$, $\bar{\gamma}(\alpha)=%0.2f$, $\varphi(\alpha)=%0.2f$' % \
            (score_n, coverage_n, fitness)
    fig, ax, im = plot_ssm_ann(S, ann, color_ann=color_ann, Fs=Fs, cmap=cmap,
                               figsize=figsize, fontsize=fontsize,
                               xlabel=r'$\alpha=[%d:%d]$' % (seg[0], seg[-1]), ylabel=ylabel, title=title)
    plot_path_family(ax[0, 0], path_family, Fs=Fs, x_offset=seg[0])
    return fig, ax, im


def visualize_scape_plot(SP, Fs=1, ax=None, figsize=(4, 3), title='',
                         xlabel='Center (seconds)', ylabel='Length (seconds)', interpolation='nearest'):
    """Visualize scape plot

    Notebook: C4/C4S3_ScapePlot.ipynb

    Args:
        SP: Scape plot data (encodes as start-duration matrix)
        Fs: Sampling rate (Default value = 1)
        ax: Used axes (Default value = None)
        figsize: Figure size (Default value = (4, 3))
        title: Title of figure (Default value = '')
        xlabel: Label for x-axis (Default value = 'Center (seconds)')
        ylabel: Label for y-axis (Default value = 'Length (seconds)')
        interpolation: Interpolation value for imshow (Default value = 'nearest')

    Returns:
        fig: Handle for figure
        ax: Handle for axes
        im: Handle for imshow
    """
    fig = None
    if ax is None:
        fig = plt.figure(figsize=figsize)
        ax = plt.gca()
    N = SP.shape[0]
    SP_vis = np.zeros((N, N))
    for length_minus_one in range(N):
        for start in range(N-length_minus_one):
            center = start + length_minus_one//2
            SP_vis[length_minus_one, center] = SP[length_minus_one, start]

    extent = np.array([-0.5, (N-1)+0.5, -0.5, (N-1)+0.5]) / Fs
    im = plt.imshow(SP_vis, cmap='hot_r', aspect='auto', origin='lower', extent=extent, interpolation=interpolation)
    x = np.asarray(range(N))
    x_half_lower = x/2
    x_half_upper = x/2 + N/2 - 1/2
    plt.plot(x_half_lower/Fs, x/Fs, '-', linewidth=3, color='black')
    plt.plot(x_half_upper/Fs, np.flip(x, axis=0)/Fs, '-', linewidth=3, color='black')
    plt.plot(x/Fs, np.zeros(N)/Fs, '-', linewidth=3, color='black')
    plt.xlim([0, (N-1) / Fs])
    plt.ylim([0, (N-1) / Fs])
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    plt.tight_layout()
    plt.colorbar(im, ax=ax)
    return fig, ax, im


# @jit(nopython=True)
def compute_fitness_scape_plot(S):
    """Compute scape plot for fitness and other measures

    Notebook: C4/C4S3_ScapePlot.ipynb

    Args:
        S (np.ndarray): Self-similarity matrix

    Returns:
        SP_all (np.ndarray): Vector containing five different scape plots for five measures
            (fitness, score, normalized score, coverage, normlized coverage)
    """
    N = S.shape[0]
    SP_fitness = np.zeros((N, N))
    SP_score = np.zeros((N, N))
    SP_score_n = np.zeros((N, N))
    SP_coverage = np.zeros((N, N))
    SP_coverage_n = np.zeros((N, N))

    for length_minus_one in range(N):
        for start in range(N-length_minus_one):
            S_seg = S[:, start:start+length_minus_one+1]
            D, score = libfmp.c4.compute_accumulated_score_matrix(S_seg)
            path_family = libfmp.c4.compute_optimal_path_family(D)
            fitness, score, score_n, coverage, coverage_n, path_family_length = libfmp.c4.compute_fitness(
                path_family, score, N)
            SP_fitness[length_minus_one, start] = fitness
            SP_score[length_minus_one, start] = score
            SP_score_n[length_minus_one, start] = score_n
            SP_coverage[length_minus_one, start] = coverage
            SP_coverage_n[length_minus_one, start] = coverage_n
    SP_all = [SP_fitness, SP_score, SP_score_n, SP_coverage, SP_coverage_n]
    return SP_all


def seg_max_sp(SP):
    """Return segment with maximal value in SP

    Notebook: C4/C4S3_ScapePlot.ipynb

    Args:
        SP (np.ndarray): Scape plot

    Returns:
        seg (tuple): Segment ``(start_index, end_index)``
    """
    N = SP.shape[0]
    # value_max = np.max(SP)
    arg_max = np.argmax(SP)
    ind_max = np.unravel_index(arg_max, [N, N])
    seg = [ind_max[1], ind_max[1]+ind_max[0]]
    return seg


def plot_seg_in_sp(ax, seg, S=None, Fs=1):
    """Plot segment and induced segements as points in SP visualization

    Notebook: C4/C4S3_ScapePlot.ipynb

    Args:
        ax: Axis for image
        seg: Segment ``(start_index, end_index)``
        S: Self-similarity matrix (Default value = None)
        Fs: Sampling rate (Default value = 1)
    """
    if S is not None:
        S_seg = S[:, seg[0]:seg[1]+1]
        D, score = libfmp.c4.compute_accumulated_score_matrix(S_seg)
        path_family = libfmp.c4.compute_optimal_path_family(D)
        segment_family, coverage = libfmp.c4.compute_induced_segment_family_coverage(path_family)
        length = segment_family[:, 1] - segment_family[:, 0] + 1
        center = segment_family[:, 0] + length//2
        ax.scatter(center/Fs, length/Fs, s=64, c='white', zorder=9999)
        ax.scatter(center/Fs, length/Fs, s=16, c='lime', zorder=9999)
    length = seg[1] - seg[0] + 1
    center = seg[0] + length//2
    ax.scatter(center/Fs, length/Fs, s=64, c='white', zorder=9999)
    ax.scatter(center/Fs, length/Fs, s=16, c='blue', zorder=9999)


def plot_sp_ssm(SP, seg, S, ann, color_ann=[], title='', figsize=(5, 4)):
    """Visulization of SP and SSM

    Notebook: C4/C4S3_ScapePlot.ipynb

    Args:
        SP: Scape plot
        seg: Segment ``(start_index, end_index)``
        S: Self-similarity matrix
        ann: Annotation
        color_ann: color scheme used for annotations (Default value = [])
        title: Title of figure (Default value = '')
        figsize: Figure size (Default value = (5, 4))
    """
    float_box = libfmp.b.FloatingBox()
    fig, ax, im = visualize_scape_plot(SP, figsize=figsize, title=title,
                                       xlabel='Center (frames)', ylabel='Length (frames)')
    plot_seg_in_sp(ax, seg, S)
    float_box.add_fig(fig)

    penalty = np.min(S)
    cmap_penalty = libfmp.c4.colormap_penalty(penalty=penalty)
    fig, ax, im = libfmp.c4.plot_ssm_ann_optimal_path_family(
        S, ann, seg, color_ann=color_ann, fontsize=8, cmap=cmap_penalty, figsize=(4, 4),
        ylabel='Time (frames)')
    float_box.add_fig(fig)
    float_box.show()


def check_segment(seg, S):
    """Prints properties of segments with regard to SSM ``S``

    Notebook: C4/C4S3_ScapePlot.ipynb

    Args:
        seg (tuple): Segment ``(start_index, end_index)``
        S (np.ndarray): Self-similarity matrix

    Returns:
         path_family (list): Optimal path family
    """
    N = S.shape[0]
    S_seg = S[:, seg[0]:seg[1]+1]
    D, score = libfmp.c4.compute_accumulated_score_matrix(S_seg)
    path_family = libfmp.c4.compute_optimal_path_family(D)
    fitness, score, score_n, coverage, coverage_n, path_family_length = libfmp.c4.compute_fitness(
                path_family, score, N)
    segment_family, coverage2 = libfmp.c4.compute_induced_segment_family_coverage(path_family)
    print('Segment (alpha):', seg)
    print('Length of segment:', seg[-1]-seg[0]+1)
    print('Length of feature sequence:', N)
    print('Induced segment path family:\n', segment_family)
    print('Fitness: %0.10f' % fitness)
    print('Score: %0.10f' % score)
    print('Normalized score: %0.10f' % score_n)
    print('Coverage: %d, %d' % (coverage, coverage2))
    print('Normalized coverage: %0.10f' % coverage_n)
    print('Length of all paths of family: %d' % path_family_length)
    return path_family
