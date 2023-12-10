from data_parser import VelParser
from entropy_compressor import HuffmanCompressor
from quantizator import Quantizator
from wavelet_compressor import WaveletCompressor

import numpy as np
from typing import Dict, List
from pathlib import Path
from pprint import pprint

COMPRESSED_PATH = 'compressed'


class Compressor:
    def __init__(self):
        self.entropy_compressor = HuffmanCompressor()
        self.quantizator = Quantizator()
        self.wavelet_compressor = WaveletCompressor()
        self.parser = VelParser()
    
    def total_compress(self, data: np.ndarray) -> List[np.ndarray | np.float64]:
        wavelet_compressed = self.wavelet_compressor.wavelet_transform(data)
        pprint(wavelet_compressed)
        quantized, q_step, min_val = self.quantizator.uniform_quantization(wavelet_compressed)
        entropy_compressed = self.entropy_compressor.huffman_encode_ndarray(quantized)
        return [entropy_compressed, q_step, min_val]

    def save_mesh(self, mesh_data: Dict[str, np.ndarray]) -> None:
        # faces, boundary, neighbour, owner, pts = mesh_data.values()
        with open(COMPRESSED_PATH + '/' + 'mesh', mode='w+') as f:
            for (_, value) in list(mesh_data.items()):
                print(value)
                compressed_ndarray, q_step, min_val = self.total_compress(value)
                f.write(f'{q_step} {min_val}\n')
                f.write(str(compressed_ndarray) + '\n')

    def compress_and_save(self, model: str, base_path: str, dim: str):
        vels = list(map(str, sorted(self.parser.get_vels(
            model=model,
            base_path=base_path,
            dimpath=dim,
        ), key=lambda x: str(x))))
        mesh_data = self.parser.get_mesh_data(
            path_to_vel=vels[0],
        )
        self.save_mesh(mesh_data=mesh_data)

        for vel in vels:
            data = self.parser.get_vel_data(
                path_to_vel=vel,
            )
            # print(data.keys())
            # print(data['0.1'])
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           
            
            

    def compress_data(self):
        pass
        



compressor = Compressor()
compressor.compress_and_save(
    model='data_wage',
    base_path='data',
    dim='low_dim',
)