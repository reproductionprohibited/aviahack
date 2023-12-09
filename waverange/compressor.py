from functools import wraps
from pathlib import Path
from typing import Union, List, Dict
from multiprocessing import Pool
from pprint import pprint
import sys

import numpy as np
import pywt

# from data_parser import VelParser


class WaveRangeCompressor:
    def __init__(self, threshold: float = 0.1):
        self.threshold = threshold
        # self.velparser = VelParser()
    
    def compress(self, data: np.ndarray) -> np.ndarray:
        pass

    def get_size(self, data: np.ndarray):
        return sys.getsizeof(data)

    def wavelet_transform(self, data, level=1):
        # Perform the nD CDF 9/7 wavelet transform
        coeffs = pywt.wavedecn(data, 'bior2.2', level=level)
        return coeffs

    def inverse_wavelet_transform(self, coeffs):
        # Perform the inverse nD CDF 9/7 wavelet transform
        reconstructed_data = pywt.waverecn(coeffs, 'bior2.2')

        return reconstructed_data

    def mean_squared_error(self, image1, image2):
        return np.mean((image1 - image2)**2)

    def quantize_coefficient(self, coeff, threshold, step):
        if coeff > threshold / 2:
            p = np.floor((coeff - threshold / 2) / step) + 1
        elif -threshold / 2 <= coeff <= threshold / 2:
            p = 0
        else:
            p = np.floor((coeff + threshold / 2) / step) - 1
        return int(p)

    def dequantize_coefficient(self, coeff, threshold, step):
        if coeff > 0:
            a_prime = (coeff - 0.5) * step + threshold / 2
        elif coeff == 0:
            a_prime = 0
        else:
            a_prime = (coeff + 0.5) * step - threshold / 2

        return a_prime
        

    def quantize_coefficients(self, coefficients, threshold, step):
        vectorized_quantize = np.vectorize(self.quantize_coefficient)
        return vectorized_quantize(coefficients, threshold, step)

    def dequantize_coefficients(self, coefficients, threshold, step):
        vectorized_dequantize = np.vectorize(self.dequantize_coefficient)
        return vectorized_dequantize(coefficients, threshold, step)


image_data = np.random.randn(100, 100, 100)
print(np.max(image_data))
print(np.min(image_data))
compressor = WaveRangeCompressor()

wavelet_coeffs = compressor.wavelet_transform(image_data, level=1)

reconstructed_image = compressor.inverse_wavelet_transform(wavelet_coeffs)
quantized_coeffs = compressor.quantize_coefficients(wavelet_coeffs, 0.1, 8)
dequantized_coeffs = compressor.dequantize_coefficients(quantized_coeffs, 0.1, 8)

print(compressor.mean_squared_error(image_data, reconstructed_image))
# print(compressor.mean_squared_error(image_data, reconstructed_image_2))
print(compressor.get_size(image_data))
print(compressor.get_size(wavelet_coeffs))
# print(compressor.get_size(quantized_coeffs))