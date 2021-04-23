"""
Module: libfmp.b.b_test_module
Author: Meinard Mueller
License: The MIT license, https://opensource.org/licenses/MIT

This file is part of the FMP Notebooks (https://www.audiolabs-erlangen.de/FMP).
"""

string = 'This is a test function'
a, b, c = 1, 2, 3


def add(a, b=0, c=0):
    """Function to add three numbers

    | Notebook: B/B_libfmp.ipynb and
    | Notebook: B/B_PythonBasics.ipynb

    Args:
        a (float): First number
        b (float): Second number (default: 0)
        c (float): Third number (default: 0)

    Returns:
        d (float): Sum
    """
    d = a + b + c
    print('Addition: ', a, ' + ', b, ' + ', c, ' = ', d)
    return d
