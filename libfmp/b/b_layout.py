"""
Module: libfmp.b.b_layout
Author: Frank Zalkow
License: The MIT license, https://opensource.org/licenses/MIT

This file is part of the FMP Notebooks (https://www.audiolabs-erlangen.de/FMP)
"""
import io
import uuid
import base64
from matplotlib import pyplot as plt
from IPython.display import HTML, display


class FloatingBox:
    """Floating box for matplotlib plots. The added figures are part of a floating box.

    Inspired by https://stackoverflow.com/a/49566213/2812618

    Attributes:
        html: The HTML string
    """
    def __init__(self, align='middle'):
        """Initializes a FloatingBox object

        Args:
            align: Vertical align of elements inside floating box, usually 'top', 'middle', or 'bottom'.
                Also see https://www.w3schools.com/cssref/pr_pos_vertical-align.asp
        """
        self.class_name = f'floating-box-fmp-{uuid.uuid4()}'
        self.html = f"""
        <style>
        .{self.class_name} {{
        display: inline-block;
        margin: 10px;
        vertical-align: {align};
        }}
        </style>
        """

    def add_fig(self, fig):
        """Saves a PNG representation of a matplotlib figure.

        Args:
            fig: A matplotlib figure
        """
        Bio = io.BytesIO()  # bytes buffer for the plot
        fig.canvas.print_png(Bio)  # make a png of the plot in the buffer

        # encode the bytes as string using base 64
        img = base64.b64encode(Bio.getvalue()).decode()
        self.html += (
            f'<div class="{self.class_name}">' +
            f'<img src="data:image/png;base64,{img}\n">' +
            '</div>')

        plt.close(fig)

    def add_html(self, html):
        """Add HTML to floating box.

        Args:
            html: HTML string
        """

        self.html += (
            f'<div class="{self.class_name}">' +
            f'{html}' +
            '</div>')

    def show(self):
        """Display the accumulated HTML"""
        display(HTML(self.html))
