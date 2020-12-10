"""
Module: libfmp.c3.c3s2_dtw_plot
Author: Frank Zalkow, Meinard Mueller
License: The MIT license, https://opensource.org/licenses/MIT

This file is part of the FMP Notebooks (https://www.audiolabs-erlangen.de/FMP)
"""
import numpy as np
from matplotlib import pyplot as plt
import libfmp.b


def plot_matrix_with_points(C, P=np.empty((0, 2)), color='r', marker='o', linestyle='', **kwargs):
    """Compute the cost matrix of two feature sequences

    Notebook: C3/C3S2_DTWbasic.ipynb

    Args:
        C: Matrix to be plotted
        P: List of index pairs, to be visualized on the matrix
        color: The color of the line plot
            See https://matplotlib.org/users/colors.html
        marker: The marker of the line plot
            See https://matplotlib.org/3.1.0/api/markers_api.html
        linestyle: The line-style of the line plot
            See https://matplotlib.org/gallery/lines_bars_and_markers/line_styles_reference.html
        **Kwargs: Arguments for `libfmp.b.plot_matrix`

    Returns:
        im: The image plot
        line: The line plot
    """

    fig, ax, im = libfmp.b.plot_matrix(C, **kwargs)
    line = ax[0].plot(P[:, 1], P[:, 0], marker=marker, color=color, linestyle=linestyle)

    if fig is not None:
        plt.tight_layout()

    return fig, im, line
