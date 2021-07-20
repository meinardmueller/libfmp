# libfmp

This repository contains the Python package libfmp. This package goes hand in hand with the FMP Notebooks, a collection of educational material for teaching and learning Fundamentals of Music Processing (FMP) with a particular focus on the audio domain. For detailed explanations and example appliciations of the libfmp-functions we refer to the FMP Notebooks:

https://audiolabs-erlangen.de/FMP

The FMP notebooks also contain a dedicated notebook for libfmp:

https://www.audiolabs-erlangen.de/resources/MIR/FMP/B/B_libfmp.html

There is also an API documentation for libfmp:

https://meinardmueller.github.io/libfmp

If you use the package libfmp, please consider the following references.

## References

Meinard Müller and Frank Zalkow. [libfmp: A Python Package for Fundamentals of Music Processing.](https://joss.theoj.org/papers/10.21105/joss.03326) Journal of Open Source Software (JOSS), 6(63), 2021.

Meinard Müller and Frank Zalkow. [FMP Notebooks: Educational Material for Teaching and Learning Fundamentals of Music Processing.](https://archives.ismir.net/ismir2019/paper/000069.pdf) Proceedings of the International Conference on Music Information Retrieval (ISMIR), pp. 573&ndash;580, Delft, The Netherlands, 2019.

Meinard Müller. [Fundamentals of Music Processing &ndash; Using Python and Jupyter Notebooks.](http://www.music-processing.de/) Springer Verlag, 2nd edition, 2021.

Meinard Müller. [An Educational Guide Through the FMP Notebooks for Teaching and Learning Fundamentals of Music Processing.](https://www.mdpi.com/2624-6120/2/2/18) Signals, 2(2): 245&ndash;285, 2021.

## Statement of Need

The libfmp package bundles core concepts from the music information retrieval (MIR) field in the form of well-documented and easy-to-use Python functions. It is designed to aid students with the transition from being learners (e.g., studying the FMP notebooks) to becoming researchers by providing proper software support for building and experimenting with complex MIR pipelines. Going beyond and complementing existing Python packages (such as librosa), the libfmp package contains (previously unpublished) reference implementations of MIR algorithms from the literature and new Python implementations of previously published MATLAB toolboxes. The functionality of libfmp addresses diverse MIR tasks such as tuning estimation, music structure analysis, audio thumbnailing, chord recognition, tempo estimation, beat and local pulse tracking, fragment-level music retrieval, and audio decomposition.

## Installing

With Python >= 3.6, you can install libfmp using the Python package manager pip:

```
pip install libfmp
```

## Contributing

The libfmp-package has been developed in the context of the FMP notebooks. Being an integral part, all libfmp-functions need to manually synchronized with text passages, explanations, and the code in the FMP notebooks. Of course, we are happy for suggestions and contributions. However, to facilitate the synchronization, we would be grateful for either directly contacting us via email (meinard.mueller@audiolabs-erlangen.de) or for creating [an issue](https://github.com/meinardmueller/libfmp/issues) in our GitHub repository. Please do not submit a pull request without prior consultation with us.

If you want to report an issue with libfmp or seek support, please use the same communication channels (email or GitHub issue).

## Tests

The functions of libmfp are also covered in the [FMP notebooks](https://audiolabs-erlangen.de/FMP). There, you find several test cases for the functions, showing typical input-output behaviors. Beyond these tests, the FMP notebooks offer extensive explanations of these functions. Thus, we consider FMP as a replacement for conventional unit tests.

Furthermore, we provide a small script that tests one function of each subpackage from libfmp. Rather than covering the full functionality of libfmp, it only verifies the correct import structure within the libfmp package.

There are two options for executing the test script. The first is just to run the script, which results in no output if there are no errors.

```
python test_examples.py
```

The second option is to use [pytest](https://pytest.org), which results in a more instructive output. pytest is available when installing libfmp with the extra requirements for testing.

```
pip install 'libfmp[tests]'
pytest test_examples.py
```

## Acknowledgements

The main authors of libfmp, Meinard Müller and Frank Zalkow, are associated with the International Audio Laboratories Erlangen, which are a joint institution of the Friedrich-Alexander-Universität Erlangen-Nürnberg (FAU) and Fraunhofer Institute for Integrated Circuits IIS. We thank the German Research Foundation (DFG) for various research grants that allow us for conducting fundamental research in music processing. Furthermore, we thank the various people who have contributed to libfmp with code and suggestions. In particular, we want to thank (in alphabetic order) Stefan Balke, Michael Krause, Patricio Lopez-Serrano, Julian Reck, Sebastian Rosenzweig, Angel Villar-Corrales, Christof Weiß, and Tim Zunner.
