---
title: 'libfmp: A Python Package for Fundamentals of Music Processing'
tags:
  - Python
  - Music information retrieval
authors:
  - name: Meinard Müller
    orcid: 0000-0001-6062-7524
    affiliation: 1
  - name: Frank Zalkow
    orcid: 0000-0003-1383-4541
    affiliation: 1
affiliations:
 - name: International Audio Laboratories Erlangen
   index: 1
date: 27 April 2021
bibliography:
  - references.bib
citation_author: Müller \& Zalkow
link-citations: yes

# Optional fields if submitting to a AAS journal too, see this blog post:
# https://blog.joss.theoj.org/2018/12/a-new-collaboration-with-aas-publishing
# aas-doi: 10.3847/xxxxx <- update this with the DOI from AAS once you know it.
# aas-journal: Astrophysical Journal <- The name of the AAS journal.
---

# Summary

The revolution in music distribution, storage, and consumption has fueled tremendous interest in developing techniques and tools for organizing, structuring, retrieving, navigating, and presenting music-related data.
As a result, the academic field of music information retrieval (MIR) has matured over the last 20 years into an independent research area related to many different disciplines, including engineering, computer science, mathematics, and musicology.
In this contribution, we introduce the Python package libfmp, which provides implementations of well-established model-based algorithms for various MIR tasks (with a focus on the audio domain), including beat tracking, onset detection, chord recognition, music synchronization, version identification, music segmentation, novelty detection, and audio decomposition.
Such traditional approaches not only yield valuable baselines for modern data-driven strategies (e.g., using deep learning) but are also instructive from an educational viewpoint deepening the understanding of the MIR task and music data at hand.
Our libfmp package is inspired and closely follows conventions as introduced by librosa, which is a widely used Python library containing standardized and flexible reference implementations of many common methods in audio and music processing [@McFeeRLEMBN15_librosa_Python].
While the two packages overlap concerning basic feature extraction and MIR algorithms, libfmp contains several reference implementations of advanced music processing pipelines not yet covered by librosa (or other open-source software). Whereas the librosa package is intended to facilitate the high-level composition of basic methods into complex pipelines, a major emphasis of libfmp is on the educational side, promoting the understanding of MIR concepts by closely following the textbook on Fundamentals of Music Processing (FMP) [@Mueller15_FMP_SPRINGER].
In this way, we hope that libfmp constitutes a valuable complement to existing open-source toolboxes such as librosa while fostering education and research in MIR.

# Introduction

Thanks to the rich and challenging domain of music, the MIR field has many things to offer to signal processing and other research disciplines [@MuellerPMV19_Editorial_IEEE-SPM]. In particular, many MIR tasks can motivate application scenarios to introduce, explain, and study techniques for audio processing, time-series analysis, and information retrieval. Furthermore, the increasing availability of suitably designed software packages and freely accessible web-based frameworks have made education in MIR, signal processing, and general computer science more interactive.
Since its beginnings, the MIR community has contributed several excellent toolboxes that provide modular source code for processing and analyzing music signals. Prominent examples are
essentia [@BogdanovWGGHMRSZS13_essentia_ISMIR],
madmom [@BoeckKSKW16_madmom_ACM-MM],
Marsyas [@Tzanetakis09_MARSYAS_ACM-MM], or the
MIRtoolbox [@LartillotT07_MirToolbox_ISMIR].
These toolboxes are mainly designed for research-oriented access to audio processing, yielding code for audio feature extraction and various MIR applications. Providing modular and readable code in combination with advanced MIR pipelines, the librosa package [@McFeeRLEMBN15_librosa_Python] strives for a low barrier to entry MIR research, thus representing a prominent example of building a bridge between education and research.

Motivated by educational considerations, the FMP notebooks, which are a comprehensive collection of educational material for teaching and learning fundamentals of music processing (FMP), were recently introduced [@MuellerZ19_FMP_ISMIR]. Closely following the textbook by @Mueller15_FMP_SPRINGER, the FMP notebooks comprise detailed textbook-like explanations of central techniques and algorithms combined with Python code examples that illustrate how to implement the methods. All components, including the introductions of MIR scenarios, illustrations, sound examples, technical concepts, mathematical details, and code examples, are integrated into a unified framework based on Jupyter notebooks. Thus, the FMP notebooks provide an interactive framework for students to learn about and experiment with signal processing and MIR algorithms.
However, when students transition from being learners to becoming researchers, they may outgrow the FMP notebooks and begin developing their own DSP methods and programs [@MuellerMK_MusicEducation_IEEE-SPM]. The libfmp package is designed to aid students with this transition by providing proper software support for building and experimenting with complex MIR pipelines.
Whereas the FMP notebooks offer an introduction to fundamental concepts in MIR step by step, the libfmp package bundles the presented core concepts in the form of well-documented and easy-to-use Python functions.
While integrating and building on the librosa package, libfmp offers alternative implementations of basic concepts in signal processing (comparing and discussing differences in the FMP notebooks). As a main contribution, libfmp provides various reference implementations of previously published MIR methods, not yet covered by librosa and other publicly available Python software.

# Statement of Need

Going beyond and complementing existing Python packages, the libfmp package contains (previously unpublished) reference implementations of MIR algorithms from the literature as well as new Python implementations of previously published MATLAB toolboxes.
For example, libfmp includes the core functionality of the MATLAB Tempogram Toolbox [@GroscheM11_PLP_TASLP; @GroscheM11_TempogramToolbox_ISMIR-lateBreaking][^1] for computing various mid-level tempogram representations [@GroscheMK10_TempogramCyclic_ICASSP] and for extracting beat and local pulse information [@GroscheM11_PLP_TASLP].
Furthermore, libfmp extends the MATLAB SM Toolbox
[@MuellerJG14_SM-Toolbox_AES][^2] for computing and enhancing similarity matrices with applications to music structure analysis. In particular, it contains a Python reference implementation for audio thumbnailing as introduced by @MuellerJG13_StructureAnaylsis_IEEE-TASLP and novel functionality for generating synthetic self-similarity matrices from reference annotations.
Other examples of previously unpublished reference implementations from the literature include the computation of tempo curves [@MuellerKSEC09_TempoParametersFromRecordings_ISMIR],  the fragment-level retrieval of different music performances [@MuellerKC05_ChromaFeatures_ISMIR], and
the separation of audio signals into harmonic, percussive, and residual components [@DriedgerMD14_SeparationHP_ISMIR].
Also, libfmp contains an implementation of a novel baseline algorithm for tuning estimation.
Finally, as shown in the FMP notebooks, the functions of libfmp can be used to conduct systematic series of experiments as described in the literature for tasks such as musically informed audio decomposition [@EwertM12_ScoreInformedNMF_ICASSP] and automated chord recognition using different chroma feature types [@JiangGKM11_Chord_AES].

[^1]: <https://www.audiolabs-erlangen.de/resources/MIR/tempogramtoolbox>
[^2]: <https://www.audiolabs-erlangen.de/resources/MIR/SMtoolbox>

# Design Choices

When designing libfmp, we had different objectives in mind.
First, we tried to keep a close connection to the textbook by @Mueller15_FMP_SPRINGER and the FMP notebooks [@MuellerZ19_FMP_ISMIR], thus establishing a close relationship between theory and practice. To this end, we carefully adopted naming conventions and the code structure to match the textbook's mathematical notions and algorithmic concepts. Furthermore, we split the package libfmp into subpackages (called B, C1, C2, \dots, C8) corresponding to the textbook's chapters. Furthermore, the docstring of each libfmp function specifies the FMP notebook where the function is introduced (typically in a step-by-step fashion interleaved with additional explanations).
Second, we followed many of the design principles suggested by librosa [@McFeeRLEMBN15_librosa_Python] to keep the entry barrier for students and researchers who may not be programming experts low. In particular, the programming style is kept explicit and straightforward with a flat, functional hierarchy.
Third, based on expert knowledge, we specified a meaningful variable preset within each function, which allows users to experiment with the code immediately.

The code of libmfp is hosted in a publicly available GitHub repository.[^3]
We also provide an API documentation for the libfmp functions,[^4] which complements the educational step-by-step explanations given in the FMP notebooks.[^5]
Finally, we included the libfmp package into the Python package index PyPi, such that libfmp can be installed with the standard Python package manager pip.[^6]

[^3]: <https://github.com/meinardmueller/libfmp>
[^4]: <https://meinardmueller.github.io/libfmp>
[^5]: <https://www.audiolabs-erlangen.de/FMP>
[^6]: <https://pypi.org/project/libfmp>


# Acknowledgements

The libfmp package builds on results, material, and insights that have been obtained in close collaboration with different people. We would like to express our gratitude to former and current students, collaborators, and colleagues who have influenced and supported us in creating this package, including
Vlora Arifi-Müller,
Stefan Balke,
Michael Krause,
Brian McFee,
Sebastian Rosenzweig,
Fabian-Robert Stöter,
Steve Tjoa,
Angel Villar-Corrales,
Christof Weiß, and
Tim Zunner.
We also thank the German Research Foundation (DFG) for various research grants that allowed us for conducting fundamental research in music processing (in particular, DFG-MU 2686/11-1, DFG-MU 2686/12-1).
The International Audio Laboratories Erlangen are a joint institution of the Friedrich-Alexander-Universität Erlangen-Nürnberg (FAU) and Fraunhofer Institute for Integrated Circuits IIS.

# References
