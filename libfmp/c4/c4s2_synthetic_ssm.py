"""
Module: libfmp.c4.c4s2_synthetic_ssm
Author: Meinard Müller, Tim Zunner
License: The MIT license, https://opensource.org/licenses/MIT

This file is part of the FMP Notebooks (https://www.audiolabs-erlangen.de/FMP)
"""
import numpy as np
import scipy.ndimage


def generate_ssm_from_annotation(ann, label_ann=None, score_path=1.0, score_block=0.5, main_diagonal=True,
                                 smooth_sigma=0.0, noise_power=0.0):
    """Generation of a SSM

    Notebook: C4/C4S2_SSM-Synthetic.ipynb

    Args:
        ann (list): Description of sections (see explanation above)
        label_ann (dict): Specification of property (path, block relation) (Default value = None)
        score_path (float): SSM values for occurring paths (Default value = 1.0)
        score_block (float): SSM values of blocks covering the same labels (Default value = 0.5)
        main_diagonal (bool): True if a filled main diagonal should be enforced (Default value = True)
        smooth_sigma (float): Standard deviation of a Gaussian smoothing filter.
            filter length is 4*smooth_sigma (Default value = 0.0)
        noise_power (float): Variance of additive white Gaussian noise (Default value = 0.0)

    Returns:
        S (np.ndarray): Generated SSM
    """
    N = ann[-1][1] + 1
    S = np.zeros((N, N))

    if label_ann is None:
        all_labels = [s[2] for s in ann]
        labels = list(set(all_labels))
        label_ann = {l: [True, True] for l in labels}

    for s in ann:
        for s2 in ann:
            if s[2] == s2[2]:
                if (label_ann[s[2]])[1]:
                    S[s[0]:s[1]+1, s2[0]:s2[1]+1] = score_block

                if (label_ann[s[2]])[0]:
                    length_1 = s[1] - s[0] + 1
                    length_2 = s2[1] - s2[0] + 1

                    if length_1 >= length_2:
                        scale_fac = length_2 / length_1
                        for i in range(s[1] - s[0] + 1):
                            S[s[0]+i, s2[0]+int(i*scale_fac)] = score_path
                    else:
                        scale_fac = length_1 / length_2
                        for i in range(s2[1] - s2[0] + 1):
                            S[s[0]+int(i*scale_fac), s2[0]+i] = score_path
    if main_diagonal:
        for i in range(N):
            S[i, i] = score_path
    if smooth_sigma > 0:
        S = scipy.ndimage.gaussian_filter(S, smooth_sigma)
    if noise_power > 0:
        S = S + np.sqrt(noise_power) * np.random.randn(S.shape[0], S.shape[1])
    return S
