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

    Notebook: B/B_libfmp.ipynb

    Args:
        a: first number
        b: second number (default: 0)
        c: third number (default: 0)

    Returns:
        Sum of a, b and c
    """
    d = a + b + c
    print('Addition: ', a, ' + ', b, ' + ', c, ' = ', d)
    return d
