from functools import wraps
from pathlib import Path
from typing import Union, List, Dict
import os
import json
from multiprocessing import Pool
from pprint import pprint

import numpy as np
import openfoamparser_mai as Ofpp
import pyvista


class Compressor:
    def wavelet_cdf97(
        self, data: np.ndarray,
    ) -> np.ndarray:
        pass

    def quantization(
        self, data: np.ndarray,
    ) -> np.ndarray:
        pass

    def entropy_encoding(
        self, data: np.ndarray,
    ) -> np.ndarray:
        pass


class Decompressor:
    def inverse_wavelet_cdf97(
        self, data: np.ndarray,
    ) -> np.ndarray:
        pass

    def inverse_quantization(
        self, data: np.ndarray,
    ) -> np.ndarray:
        pass

    def entropy_decoding(
        self, data: np.ndarray,
    ) -> np.ndarray:
        pass
