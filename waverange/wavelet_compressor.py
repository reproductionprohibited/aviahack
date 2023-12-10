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


# # image_data = np.array([e, pi, e, pi, e, pi, e, pi], dtype=np.float64)
# image_data = np.random.rand(5, 5, 5) * 1000000
# # print(np.max(image_data))
# # print(np.min(image_data))
# compressor = WaveletCompressor()

# wavelet_coeffs = compressor.wavelet_transform(image_data, level=2)

# reconstructed_image = compressor.inverse_wavelet_transform(wavelet_coeffs)

# # print(compressor.mean_squared_error(image_data, reconstructed_image))
# print(compressor.get_size(image_data))
# print(compressor.get_size(wavelet_coeffs))
