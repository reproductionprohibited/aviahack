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

    def quantize_data(self, data):
        arr = np.asarray(data)
        



image_data = np.random.randn(1, 1, 1)
print(np.max(image_data))
print(np.min(image_data))
compressor = WaveRangeCompressor()

wavelet_coeffs = compressor.wavelet_transform(image_data, level=1)

# quantization_factor = 10  # Adjust this factor based on your requirements
# quantized_coeffs = compressor.quantize_data(wavelet_coeffs, quantization_factor)
# original_coeffs = compressor.dequantize_data(quantized_coeffs, quantization_factor)

reconstructed_image = compressor.inverse_wavelet_transform(wavelet_coeffs)
# reconstructed_image_2 = compressor.inverse_wavelet_transform(quantized_coeffs)

print(compressor.mean_squared_error(image_data, reconstructed_image))
# print(compressor.mean_squared_error(image_data, reconstructed_image_2))
print(compressor.get_size(image_data))
print(compressor.get_size(wavelet_coeffs))
# print(compressor.get_size(quantized_coeffs))