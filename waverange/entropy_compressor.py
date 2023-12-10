import numpy as np
import zlib
import sys


class ZCompressor:
    def compress(self, data: np.ndarray) -> bytes:
        return zlib.compress(data.tobytes())
    
    def decompress(self, compressed: bytes) -> np.ndarray:
        return np.frombuffer(zlib.decompress(compressed), dtype=np.float64)
    
    def mse(self, start, decommpressed) -> np.float64:
        return (np.mean(start - decommpressed)**2)
