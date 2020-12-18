import os

import numpy as np
from matplotlib import pyplot as plt

import libfmp.b
import libfmp.c1
import libfmp.c2
import libfmp.c3
import libfmp.c4
import libfmp.c5
import libfmp.c6
import libfmp.c7
import libfmp.c8


def test_b():
    fn_pdf = 'test_plot_matrix.pdf'
    X = np.random.random((10, 10))

    if os.path.exists(fn_pdf):
        os.remove(fn_pdf)

    libfmp.b.plot_matrix(X)
    #print('text_b')	
    #plt.show()
    plt.savefig(fn_pdf)

    assert os.path.exists(fn_pdf)


def test_c1():
    freq69 = libfmp.c1.f_pitch(69)
    assert freq69 == 440.0


def test_c2():
    N = 512
    n = np.arange(N)
    k = 20
    f = k / N
    x = np.sin(2 * np.pi * f * n)
    X = libfmp.c2.fft(x)

    assert np.argmax(np.abs(X)) == k


def test_c3():
    freq69 = libfmp.c3.f_pitch(69, freq_ref=415.0)
    assert freq69 == 415.0


def test_c4():
    structure_annotation = [(0, 1, 'A'), (1, 2, 'B')]
    S = libfmp.c4.convert_structure_annotation(structure_annotation, Fs=10)
    assert S == [[0, 10, 'A'], [10, 20, 'B']]


def test_c5():
    X = np.zeros((12, 25))

    for root in range(12):
        # major chords
        for c in np.array([root, root+4, root+7]) % 12:
            X[c, root] = 1
        # minor chords
        for c in np.array([root, root+3, root+7]) % 12:
            X[c, 12 + root] = 1

    X[:, -1] = 1

    chord_sim, chord_max = libfmp.c5.chord_recognition_template(X, nonchord=True)
    chord_idx = np.argmax(chord_sim, axis=0)

    assert np.all(chord_idx == np.arange(25))


def test_c6():
    Fs = 22050
    beats = np.array([0.5, 1, 2]) * Fs
    tempo = libfmp.c6.beat_period_to_tempo(beats, Fs)
    assert np.all(tempo == np.array([120.0, 60.0, 30.0]))


def test_c7():
    X = np.random.random((12, 100))
    X = X / np.sqrt(np.sum(np.square(X), axis=0))
    C = libfmp.c7.cost_matrix_dot(X, X)

    assert np.allclose(np.diag(C), 0.0)


def test_c8():
    cents = libfmp.c8.hz_to_cents(880, F_ref=440)
    assert cents == 1200


if __name__ == '__main__':
    test_b()
    test_c1()
    test_c2()
    test_c3()
    test_c4()
    test_c5()
    test_c6()
    test_c7()
    test_c8()
