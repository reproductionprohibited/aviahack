from functools import wraps
from pathlib import Path
from typing import Union, List, Dict
from multiprocessing import Pool
from pprint import pprint
import sys

import numpy as np
import pywt

from math import pi
e = 2.718281828459045


class WaveletCompressor:
    def __init__(self, threshold: float = 0.1):
        self.threshold = threshold

    def get_size(self, data: np.ndarray):
        return sys.getsizeof(data)

    def wavelet_transform(self, data, level=2):
        """ Perform the nD CDF 9/7 wavelet transform """
        coeffs = pywt.wavedecn(data, 'bior2.2', level=level)
        return coeffs

    def inverse_wavelet_transform(self, coeffs):
        """ Perform the inverse nD CDF 9/7 wavelet transform """
        reconstructed_data = pywt.waverecn(coeffs, 'bior2.2')

        return reconstructed_data

    def mean_squared_error(self, image1, image2):
        return np.mean((image1 - image2) ** 2)
