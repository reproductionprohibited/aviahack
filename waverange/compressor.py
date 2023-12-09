from data_parser import VelParser
from entropy_compressor import HuffmanCompressor
from quantizator import Quantizator
from wavelet_compressor import WaveletCompressor

from pathlib import Path
import pprint


class Compressor:
    def __init__(self):
        self.entropy_compressor = HuffmanCompressor()
        self.quantizator = Quantizator()
        self.wavelet_compressor = WaveletCompressor()
        self.parser = VelParser()
    
    def compress_and_save(self, model: str, base_path: str, dim: str):
        vels = list(map(str, sorted(self.parser.get_vels(
            model=model,
            base_path=base_path,
            dimpath=dim,
        ), key=lambda x: str(x))))
        for vel in vels:
            data = self.parser.get_vel_data(
                vel=vel,
                model=model,
                base_path=base_path,
                dimpath=dim,
            )
            pprint.pprint(data)



    def compress_data(self):
        pass
        



compressor = Compressor()
compressor.compress_and_save(
    model='data_wage',
    base_path='data',
    dim='low_dim',
)