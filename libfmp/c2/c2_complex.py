"""
Module: libfmp.c2.c2_complex
Author: Meinard Müller, Frank Zalkow
License: The MIT license, https://opensource.org/licenses/MIT

This file is part of the FMP Notebooks (https://www.audiolabs-erlangen.de/FMP)
"""
import numpy as np
from matplotlib import pyplot as plt


def generate_figure(figsize=(2, 2), xlim=[0, 1], ylim=[0, 1]):
    """Generate figure for plotting complex numbers

    Notebook: C2/C2_ComplexNumbers.ipynb

    Args:
        figsize: Figure size (Default value = (2, 2))
        xlim: Limits of x-axis (Default value = [0, 1])
        ylim: Limits of y-axis (Default value = [0, 1])
    """
    plt.figure(figsize=figsize)
    plt.grid()
    plt.xlim(xlim)
    plt.ylim(ylim)
    plt.xlabel(r'$\mathrm{Re}$')
    plt.ylabel(r'$\mathrm{Im}$')


def plot_vector(c, color='k', start=0, linestyle='-'):
    """Plot arrow corresponding to difference of two complex numbers

    Notebook: C2/C2_ComplexNumbers.ipynb

    Args:
        c: Complex number
        color: Color of arrow (Default value = 'k')
        start: Complex number encoding the start position (Default value = 0)
        linestyle: Linestyle of arrow (Default value = '-')

    Returns:
        arrow (matplotlib.patches.FancyArrow): Arrow
    """
    return plt.arrow(np.real(start), np.imag(start), np.real(c), np.imag(c),
                     linestyle=linestyle, head_width=0.05, fc=color, ec=color, overhang=0.3,
                     length_includes_head=True)
