"""
Module: libfmp.b.plot
Author: Frank Zalkow, Meinard Mueller
License: The MIT license, https://opensource.org/licenses/MIT

This file is part of the FMP Notebooks (https://www.audiolabs-erlangen.de/FMP)
"""

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatch


FMP_COLORMAPS = {
    'FMP_1': np.array([[1.0, 0.5, 0.0], [0.33, 0.75, 0.96], [0.0, 1.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 1.0],
                       [1.0, 0.0, 1.0],  [0.99, 0.51, 0.71], [0.53, 0.0, 0.46], [0.56, 0.93, 0.72], [0, 0, 0.9]])
}


def plot_signal(x, Fs=1, T_coef=None, ax=None, figsize=(6, 2), xlabel='Time (seconds)', ylabel='', title='', dpi=72,
                ylim=True, **kwargs):
    """Line plot visualization of a signal, e.g. a waveform or a novelty function.

    Args:
        x: Input signal
        Fs: Sample rate (Default value = 1)
        T_coef: Time coeffients. If None, will be computed, based on Fs. (Default value = None)
        ax: The Axes instance to plot on. If None, will create a figure and axes. (Default value = None)
        figsize: Width, height in inches (Default value = (6, 2))
        xlabel: Label for x axis (Default value = 'Time (seconds)')
        ylabel: Label for y axis (Default value = '')
        title: Title for plot (Default value = '')
        dpi: Dots per inch (Default value = 72)
        ylim: True or False (auto adjust ylim or nnot) or tuple with actual ylim (Default value = True)
        **kwargs: Keyword arguments for matplotlib.pyplot.plot

    Returns:
        fig: The created matplotlib figure or None if ax was given.
        ax: The used axes.
        line: The line plot
    """
    fig = None
    if ax is None:
        fig = plt.figure(figsize=figsize, dpi=dpi)
        ax = plt.subplot(1, 1, 1)
    if T_coef is None:
        T_coef = np.arange(x.shape[0]) / Fs

    if 'color' not in kwargs:
        kwargs['color'] = 'gray'

    line = ax.plot(T_coef, x, **kwargs)

    ax.set_xlim([T_coef[0], T_coef[-1]])
    if ylim is True:
        ylim_x = x[np.isfinite(x)]
        x_min, x_max = ylim_x.min(), ylim_x.max()
        if x_max == x_min:
            x_max = x_max + 1
        ax.set_ylim([min(1.1 * x_min, 0.9 * x_min), max(1.1 * x_max, 0.9 * x_max)])
    elif ylim not in [True, False, None]:
        ax.set_ylim(ylim)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    if fig is not None:
        plt.tight_layout()

    return fig, ax, line


def plot_matrix(X, Fs=1, Fs_F=1, T_coef=None, F_coef=None, xlabel='Time (seconds)', ylabel='Frequency (Hz)',
                xlim=None, ylim=None, clim=None, title='', dpi=72,
                colorbar=True, colorbar_aspect=20.0, cbar_label='', ax=None, figsize=(6, 3), **kwargs):
    """2D raster visualization of a matrix, e.g. a spectrogram or a tempogram.

    Args:
        X: The matrix
        Fs: Sample rate for axis 1 (Default value = 1)
        Fs_F: Sample rate for axis 0 (Default value = 1)
        T_coef: Time coeffients. If None, will be computed, based on Fs. (Default value = None)
        F_coef: Frequency coeffients. If None, will be computed, based on Fs_F. (Default value = None)
        xlabel: Label for x-axis (Default value = 'Time (seconds)')
        ylabel: Label for y-axis (Default value = 'Frequency (Hz)')
        xlim: Limits for x-axis (Default value = None)
        ylim: Limits for y-axis (Default value = None)
        clim: Color limits (Default value = None)
        title: Title for plot (Default value = '')
        dpi: Dots per inch (Default value = 72)
        colorbar: Create a colorbar. (Default value = True)
        colorbar_aspect: Aspect used for colorbar, in case only a single axes is used. (Default value = 20.0)
        cbar_label: Label for colorbar (Default value = '')
        ax: Either (1.) a list of two axes (first used for matrix, second for colorbar), or (2.) a list with a single
            axes (used for matrix), or (3.) None (an axes will be created). (Default value = None)
        figsize: Width, height in inches (Default value = (6, 3))
        **kwargs: Keyword arguments for matplotlib.pyplot.imshow

    Returns:
        fig: The created matplotlib figure or None if ax was given.
        ax: The used axes.
        im: The image plot
    """
    fig = None
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=figsize, dpi=dpi)
        ax = [ax]
    if T_coef is None:
        T_coef = np.arange(X.shape[1]) / Fs
    if F_coef is None:
        F_coef = np.arange(X.shape[0]) / Fs_F

    if 'extent' not in kwargs:
        x_ext1 = (T_coef[1] - T_coef[0]) / 2
        x_ext2 = (T_coef[-1] - T_coef[-2]) / 2
        y_ext1 = (F_coef[1] - F_coef[0]) / 2
        y_ext2 = (F_coef[-1] - F_coef[-2]) / 2
        kwargs['extent'] = [T_coef[0] - x_ext1, T_coef[-1] + x_ext2, F_coef[0] - y_ext1, F_coef[-1] + y_ext2]
    if 'cmap' not in kwargs:
        kwargs['cmap'] = 'gray_r'
    if 'aspect' not in kwargs:
        kwargs['aspect'] = 'auto'
    if 'origin' not in kwargs:
        kwargs['origin'] = 'lower'
    if 'interpolation' not in kwargs:
        kwargs['interpolation'] = 'nearest'

    im = ax[0].imshow(X, **kwargs)

    if len(ax) == 2 and colorbar:
        cbar = plt.colorbar(im, cax=ax[1])
        cbar.set_label(cbar_label)
    elif len(ax) == 2 and not colorbar:
        ax[1].set_axis_off()
    elif len(ax) == 1 and colorbar:
        plt.sca(ax[0])
        cbar = plt.colorbar(im, aspect=colorbar_aspect)
        cbar.set_label(cbar_label)

    ax[0].set_xlabel(xlabel)
    ax[0].set_ylabel(ylabel)
    ax[0].set_title(title)
    if xlim is not None:
        ax[0].set_xlim(xlim)
    if ylim is not None:
        ax[0].set_ylim(ylim)
    if clim is not None:
        im.set_clim(clim)

    if fig is not None:
        plt.tight_layout()

    return fig, ax, im


def plot_chromagram(*args, chroma_yticks=np.arange(12), **kwargs):
    """Call libfmp.b.plot_matrix and sets chroma labels.

    See :func:`libfmp.b.b_plot.plot_matrix` for parameters and return values.
    """
    if 'ylabel' not in kwargs:
        kwargs['ylabel'] = 'Chroma'
    fig, ax, im = plot_matrix(*args, **kwargs)

    chroma_names = 'C C# D D# E F F# G G# A A# B'.split()
    ax[0].set_yticks(np.array(chroma_yticks))
    ax[0].set_yticklabels([chroma_names[i] for i in chroma_yticks])

    return fig, ax, im


def compressed_gray_cmap(alpha=5, N=256, reverse=False):
    """Create a logarithmically or exponentially compressed grayscale colormap.

    Args:
        alpha (float): The compression factor. If alpha > 0, it performs log compression (enhancing black colors).
            If alpha < 0, it performs exp compression (enhancing white colors).
            Raises an error if alpha = 0. (Default value = 5)
        N (int): The number of rgb quantization levels (usually 256 in matplotlib) (Default value = 256)
        reverse (bool): If False then "white to black", if True then "black to white" (Default value = False)

    Returns:
        color_wb (mpl.colors.LinearSegmentedColormap): The colormap
    """
    assert alpha != 0

    gray_values = np.log(1 + abs(alpha) * np.linspace(0, 1, N))
    gray_values /= gray_values.max()

    if alpha > 0:
        gray_values = 1 - gray_values
    else:
        gray_values = gray_values[::-1]

    if reverse:
        gray_values = gray_values[::-1]

    gray_values_rgb = np.repeat(gray_values.reshape(N, 1), 3, axis=1)
    color_wb = LinearSegmentedColormap.from_list('color_wb', gray_values_rgb, N=N)
    return color_wb


class MultiplePlotsWithColorbar():
    """Two-column layout plot, where the first column is for user-given plots and the second column
    is for colorbars if the corresponding row needs a colorbar.

    Attributes:
        axes: A list of axes for the first column.
        cbar_axes: A list of axes for the second column.
        num_plots: Number of rows, as given to init method.
    """

    def __init__(self, num_plots, figsize=(8, 4), dpi=72, cbar_ratio=0.1, height_ratios=None):
        """Create an instance of the MultiplePlotsWithColorbar class

        Args:
            num_plots: Number of plots (also number of rows)
            figsize: Figure size in dpi
            dpi: Dots per inch
            cbar_ratio: Width ratio of color bar
            height_ratios: Height ratio for rows
        """
        if height_ratios is None:
            height_ratios = [1] * num_plots

        plt.figure(figsize=figsize, dpi=dpi)
        gs = gridspec.GridSpec(num_plots, 2, width_ratios=[1, cbar_ratio], height_ratios=height_ratios)

        self.num_plots = num_plots
        self.axes = []
        self.cbar_axes = []

        for i in range(self.num_plots):
            self.axes.append(plt.subplot(gs[i, 0]))
            self.cbar_axes.append(plt.subplot(gs[i, 1]))

    def make_colorbars(self):
        """Create colorbars if the corresponding row needs a colorbar, or hides the axis in other cases."""
        for i in range(self.num_plots):
            ax_img = self.axes[i].get_images()

            if len(ax_img) == 0:
                self.cbar_axes[i].set_axis_off()
            else:
                plt.colorbar(ax_img[0], cax=self.cbar_axes[i])

        plt.tight_layout()


def color_argument_to_dict(colors, labels_set, default='gray'):
    """Create a dictionary that maps labels to colors.

    Args:
        colors: Several options: 1. string of ``FMP_COLORMAPS``, 2. string of matplotlib colormap,
            3. list or np.ndarray of matplotlib color specifications, 4. dict that assigns labels  to colors
        labels_set: List of all labels
        default: Default color, used for labels that are in labels_set, but not in colors

    Returns:
        color_dict: Dictionary that maps labels to colors
    """

    if isinstance(colors, str):
        # FMP colormap
        if colors in FMP_COLORMAPS:
            color_dict = {l: c for l, c in zip(labels_set, FMP_COLORMAPS[colors])}
        # matplotlib colormap
        else:
            cm = plt.get_cmap(colors)
            num_labels = len(labels_set)
            colors = [cm(i / (num_labels + 1)) for i in range(num_labels)]
            color_dict = {l: c for l, c in zip(labels_set, colors)}

    # list/np.ndarray of colors
    elif isinstance(colors, (list, np.ndarray, tuple)):
        color_dict = {l: c for l, c in zip(labels_set, colors)}

    # is already a dict, nothing to do
    elif isinstance(colors, dict):
        color_dict = colors

    else:
        raise ValueError('`colors` must be str, list, np.ndarray, or dict')

    for key in labels_set:
        if key not in color_dict:
            color_dict[key] = default

    return color_dict


def check_line_annotations(annot, default_label=''):
    """Check a list of annotations with time positions and labels. If label is missing, adds an default label.

    Args:
        annot: A List of the form of ``[(time_position, label), ...]``, or ``[(time_position, ), ...]``,
            or ``[time_position, ...]``
        default_label: The default label used if label is missing

    Returns:
        annot: A List of tuples in the form of ``[(time_position, label), ...]``
    """
    if isinstance(annot[0], (list, np.ndarray, tuple)):
        len_annot = len(annot[0])
        assert all(len(a) == len_annot for a in annot)
        if len_annot == 1:
            annot = [(a[0], default_label) for a in annot]

    else:
        assert isinstance(annot[0], (int, float, complex)) or np.isscalar(annot[0])
        annot = [(a, default_label) for a in annot]

    return annot


def check_segment_annotations(annot, default_label=''):
    """Check a list of segment annotations. If label is missing, adds an default label.

    Args:
        annot: A List of the form of ``[(start_position, end_position, label), ...]``, or
            ``[(start_position, end_position), ...]``
        default_label: The default label used if label is missing

    Returns:
        annot: A List of tuples in the form of ``[(start_position, end_position, label), ...]``
    """
    assert isinstance(annot[0], (list, np.ndarray, tuple))
    len_annot = len(annot[0])
    assert all(len(a) == len_annot for a in annot)
    if len_annot == 2:
        annot = [(a[0], a[1], default_label) for a in annot]

    return annot


def plot_annotation_line(annot, ax=None, label_keys=None, colors='FMP_1', figsize=(6, 1), direction='horizontal',
                         time_min=None, time_max=None, time_axis=True, nontime_axis=False, swap_time_ticks=False,
                         axis_off=False, dpi=72):
    """Create a line plot visualization for a list of annotations with time positions and labels.

    Args:
        annot: A List of tuples in the form of ``[(time_position, label), ...]``
        ax: The Axes instance to plot on. If None, will create a figure and axes. (Default value = None)
        label_keys: A dict, where the keys are the labels used in `annot`. The values are dicts, which are used as
            keyword arguments for matplotlib.pyplot.axvline or matplotlib.pyplot.axhline. (Default value = None)
        colors: Several options: 1. string of ``FMP_COLORMAPS``, 2. string of matplotlib colormap, 3. list or
            np.ndarray of matplotlib color specifications, 4. dict that assigns labels  to colors
            (Default value = 'FMP_1')
        figsize: Width, height in inches (Default value = (6, 1)
        direction: 'vertical' or 'horizontal' (Default value = 'horizontal')
        time_min: Minimal limit for time axis. If None, will be min annotation. (Default value = None)
        time_max: Maximal limit for time axis. If None, will be max from annotation. (Default value = None)
        time_axis: Display time axis ticks or not (Default value = True)
        nontime_axis: Display non-time axis ticks or not (Default value = False)
        swap_time_ticks: For horizontal: xticks up; for vertical: yticks left (Default value = False)
        axis_off: Calls ax.axis('off') (Default value = False)
        dpi: Dots per inch (Default value = 72)

    Returns:
        fig: The created matplotlib figure or None if ax was given.
        ax: The used axes.
    """

    assert direction in ['vertical', 'horizontal']
    annot = check_line_annotations(annot)

    fig = None
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=figsize, dpi=dpi)

    labels_set = sorted(set([label for pos, label in annot]))
    colors = color_argument_to_dict(colors, labels_set)

    if label_keys is None:
        label_keys = {}

    for key, value in colors.items():
        if key not in label_keys:
            label_keys[key] = {}
        if 'color' not in label_keys[key]:
            label_keys[key]['color'] = value

    for pos, label in annot:
        if direction == 'horizontal':
            ax.axvline(pos, **label_keys[label])
        else:
            ax.axhline(pos, **label_keys[label])

    if time_min is None:
        time_min = min(pos for pos, label in annot)
    if time_max is None:
        time_max = max(pos for pos, label in annot)

    if direction == 'horizontal':
        ax.set_xlim([time_min, time_max])
        if not time_axis:
            ax.set_xticks([])
        if not nontime_axis:
            ax.set_yticks([])
        if swap_time_ticks:
            ax.xaxis.tick_top()
    else:
        ax.set_ylim([time_min, time_max])
        if not time_axis:
            ax.set_yticks([])
        if not nontime_axis:
            ax.set_xticks([])
        if swap_time_ticks:
            ax.yaxis.tick_right()

    if axis_off:
        ax.axis('off')

    if fig is not None:
        plt.tight_layout()

    return fig, ax


def plot_annotation_line_overlay(*args, **kwargs):
    """Plot segment annotations as overlay.

    See :func:`libfmp.b.b_plot.plot_annotation_line` for parameters and return values.
    """
    assert 'nontime_axis' not in kwargs
    kwargs['nontime_axis'] = True
    return plot_annotation_line(*args, **kwargs)


def plot_annotation_multiline(annot, ax=None, label_keys=None, colors='FMP_1', figsize=(6, 1.5), direction='horizontal',
                              sort_labels=None, time_min=None, time_max=None, time_axis=True, swap_time_ticks=False,
                              axis_off=False, dpi=72):
    """Create a multi-line plot visualization for a list of annotations with time positions and labels.
    The multi-line plot visualization comprises a separate line plot visualization for each label category stacked
    horizontally or vertically (depending on the ``direction`` keyword).

    Args:
        annot: A List of tuples in the form of ``[(time_position, label), ...]``
        ax: The Axes instance to plot on. If None, will create a figure and axes. (Default value = None)
        label_keys: A dict, where the keys are the labels used in ``annot``. The values are dicts, which are used as
            keyword arguments for matplotlib.pyplot.axvline or matplotlib.pyplot.axhline. (Default value = None)
        colors: Several options: 1. string of ``FMP_COLORMAPS``, 2. string of matplotlib colormap, 3. list or np.ndarray
            of matplotlib color specifications, 4. dict that assigns labels  to colors (Default value = 'FMP_1')
        figsize: Width, height in inches (Default value = (6, 1.5)
        direction: 'vertical' or 'horizontal' (Default value = 'horizontal')
        sort_labels: List of labels used for sorting the line plots (Default value = None)
        time_min: Minimal limit for time axis. If None, will be min annotation. (Default value = None)
        time_max: Maximal limit for time axis. If None, will be max from annotation. (Default value = None)
        time_axis: Display time axis ticks or not (Default value = True)
        swap_time_ticks: For horizontal: xticks up; for vertical: yticks left (Default value = False)
        axis_off: Calls ax.axis('off') (Default value = False)
        dpi: Dots per inch (Default value = 72)

    Returns:
        fig: The created matplotlib figure or None if ax was given.
        ax: The used axes.
    """

    assert direction in ['vertical', 'horizontal']
    annot = check_line_annotations(annot)

    fig = None
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=figsize, dpi=dpi)

    labels_set = sorted(set([label for pos, label in annot]))
    colors = color_argument_to_dict(colors, labels_set)

    if label_keys is None:
        label_keys = {}

    for key, value in colors.items():
        if key not in label_keys:
            label_keys[key] = {}
        if 'color' not in label_keys[key]:
            label_keys[key]['color'] = value

    if sort_labels:
        sort_func = lambda x: sort_labels.index(x) if x in sort_labels else 0
    else:
        sort_func = None

    all_labels = sorted(set(label for pos, label in annot), key=sort_func)

    for i, cur_label in enumerate(all_labels):
        cur_pos = [pos for pos, label in annot if cur_label == label]

        if direction == 'horizontal':
            ax.vlines(cur_pos, i, i+1, **label_keys[cur_label])
        else:
            ax.hlines(cur_pos, i, i+1, **label_keys[cur_label])

    if time_min is None:
        time_min = min(pos for pos, label in annot)
    if time_max is None:
        time_max = max(pos for pos, label in annot)

    if direction == 'horizontal':
        ax.set_ylim([0, len(all_labels)])
        ax.set_xlim([time_min, time_max])
        for seperator in range(1, len(all_labels)):
            ax.axhline(seperator, color='k')
        ax.set_yticks(np.arange(len(all_labels)) + 0.5)
        ax.set_yticklabels(all_labels)
        if not time_axis:
            ax.set_xticks([])
        if swap_time_ticks:
            ax.xaxis.tick_top()

    else:
        ax.set_xlim([0, len(all_labels)])
        ax.set_ylim([time_min, time_max])
        for seperator in range(1, len(all_labels)):
            ax.axvline(seperator, color='k')
        ax.set_xticks(np.arange(len(all_labels)) + 0.5)
        ax.set_xticklabels(all_labels, rotation=90)
        if not time_axis:
            ax.set_yticks([])
        if swap_time_ticks:
            ax.yaxis.tick_right()

    if axis_off:
        ax.axis('off')

    if fig is not None:
        plt.tight_layout()

    return fig, ax


def plot_segments(annot, ax=None, figsize=(6, 1), direction='horizontal', colors='FMP_1', time_min=None,
                  time_max=None, nontime_min=0, nontime_max=1, time_axis=True, nontime_axis=False, time_label=None,
                  swap_time_ticks=False, edgecolor='k', axis_off=False, dpi=72, adjust_time_axislim=True,
                  adjust_nontime_axislim=True, alpha=None, print_labels=True, label_ticks=False, **kwargs):
    """Create a segment plot visualization for a list of annotations with start time positions, end time
    positions, and labels. The segment plot visualization shows a colored box for each segment.

    Args:
        annot: A List of tuples in the form of ``[(start_position, end_position, label), ...]``
        ax: The Axes instance to plot on. If None, will create a figure and axes. (Default value = None)
        figsize: Width, height in inches (Default value = (6, 1)
        direction: 'vertical' or 'horizontal' (Default value = 'horizontal')
        colors: Several options: 1. string of ``FMP_COLORMAPS``, 2. string of matplotlib colormap, 3. list or np.ndarray
            of matplotlib color specifications, 4. dict that assigns labels  to colors (Default value = 'FMP_1')
        time_min: Minimal limit for time axis. If None, will be min annotation. (Default value = None)
        time_max: Maximal limit for time axis. If None, will be max from annotation. (Default value = None)
        nontime_min: Minimal limit for non-time axis. (Default value = 0)
        nontime_max: Maximal limit for non-time axis. (Default value = 1)
        time_axis: Display time axis ticks or not (Default value = True)
        nontime_axis: Display non-time axis ticks or not (Default value = False)
        time_label: Label for time axes (Default value = None)
        swap_time_ticks: For horizontal: xticks up; for vertical: yticks left (Default value = False)
        edgecolor: Color for edgelines of segment box (Default value = 'k')
        axis_off: Calls ax.axis('off') (Default value = False)
        dpi: Dots per inch (Default value = 72)
        adjust_time_axislim: Adjust time-axis. Usually True for plotting on standalone axes and False for
            overlay plotting (Default value = True)
        adjust_nontime_axislim: Adjust non-time-axis. Usually True for plotting on standalone axes and False for
            overlay plotting (Default value = True)
        alpha: Alpha value for rectangle (Default value = None)
        print_labels: Print labels inside Rectangles (Default value = True)
        label_ticks: Print labels as ticks (Default value = False)
        **kwargs:

    Returns:
        fig: The created matplotlib figure or None if ax was given.
        ax: The used axes.
    """
    assert direction in ['vertical', 'horizontal']
    annot = check_segment_annotations(annot)

    if 'color' not in kwargs:
        kwargs['color'] = 'k'
    if 'weight' not in kwargs:
        kwargs['weight'] = 'bold'
        # kwargs['weight'] = 'normal'
    if 'fontsize' not in kwargs:
        kwargs['fontsize'] = 12
    if 'ha' not in kwargs:
        kwargs['ha'] = 'center'
    if 'va' not in kwargs:
        kwargs['va'] = 'center'

    if colors is None:
        colors = 'FMP_1'

    fig = None
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=figsize, dpi=dpi)

    labels_set = sorted(set([label for start, end, label in annot]))
    colors = color_argument_to_dict(colors, labels_set)

    nontime_width = nontime_max - nontime_min
    nontime_middle = nontime_min + nontime_width / 2
    all_time_middles = []

    for start, end, label in annot:
        time_width = end - start
        time_middle = start + time_width / 2
        all_time_middles.append(time_middle)

        if direction == 'horizontal':
            rect = mpatch.Rectangle((start, nontime_min), time_width, nontime_width,
                                    facecolor=colors[label], edgecolor=edgecolor, alpha=alpha)
            ax.add_patch(rect)
            if print_labels:
                ax.annotate(label, (time_middle, nontime_middle), **kwargs)
        else:
            rect = mpatch.Rectangle((nontime_min, start), nontime_width, time_width,
                                    facecolor=colors[label], edgecolor=edgecolor, alpha=alpha)
            ax.add_patch(rect)
            if print_labels:
                ax.annotate(label, (nontime_middle, time_middle), **kwargs)

    if time_min is None:
        time_min = min(start for start, end, label in annot)
    if time_max is None:
        time_max = max(end for start, end, label in annot)

    if direction == 'horizontal':
        if adjust_time_axislim:
            ax.set_xlim([time_min, time_max])
        if adjust_nontime_axislim:
            ax.set_ylim([nontime_min, nontime_max])
        if not nontime_axis:
            ax.set_yticks([])
        if not time_axis:
            ax.set_xticks([])
        if swap_time_ticks:
            ax.xaxis.tick_top()
        if time_label:
            ax.set_xlabel(time_label)
        if label_ticks:
            ax.set_xticks(all_time_middles)
            ax.set_xticklabels([label for start, end, label in annot])

    else:
        if adjust_time_axislim:
            ax.set_ylim([time_min, time_max])
        if adjust_nontime_axislim:
            ax.set_xlim([nontime_min, nontime_max])
        if not nontime_axis:
            ax.set_xticks([])
        if not time_axis:
            ax.set_yticks([])
        if swap_time_ticks:
            ax.yaxis.tick_right()
        if time_label:
            ax.set_ylabel(time_label)
        if label_ticks:
            ax.set_yticks(all_time_middles)
            ax.set_yticklabels([label for start, end, label in annot])

    if axis_off:
        ax.axis('off')

    if fig is not None:
        plt.tight_layout()

    return fig, ax


def plot_segments_overlay(*args, **kwargs):
    """Plot segment annotations as overlay.

    See :func:`libfmp.b.b_plot.plot_segments` for parameters and return values.
    """
    assert 'ax' in kwargs
    ax = kwargs['ax']

    if 'adjust_time_axislim' not in kwargs:
        kwargs['adjust_time_axislim'] = False
    if 'adjust_nontime_axislim' not in kwargs:
        kwargs['adjust_nontime_axislim'] = False
    if 'alpha' not in kwargs:
        kwargs['alpha'] = 0.3
    if 'edgecolor' not in kwargs:
        kwargs['edgecolor'] = None
    if 'nontime_axis' not in kwargs:
        kwargs['nontime_axis'] = True

    if 'direction' in kwargs and kwargs['direction'] == 'vertical':
        kwargs['nontime_min'], kwargs['nontime_max'] = ax.get_xlim()
    else:
        kwargs['nontime_min'], kwargs['nontime_max'] = ax.get_ylim()

    fig, ax = plot_segments(*args, **kwargs)

    return fig, ax
