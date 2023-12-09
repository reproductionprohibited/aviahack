import sys

import numpy as np
from heapq import heappush, heappop, heapify
from collections import defaultdict

from math import pi
e = 2.718281828459045


class HuffmanCompressor:
    class HuffmanNode:
        def __init__(self, symbol=None, freq=None):
            self.symbol = symbol
            self.freq = freq
            self.left = None
            self.right = None

        def __lt__(self, other):
            return self.freq < other.freq

    def __init__(self):
        self.huffman_tree = None
        self.code_table = None
        self.original_shape = None

    def flatten_ndarray(self, ndarray):
        return ndarray.flatten()

    def reshape_to_ndarray(self, flat_array):
        return flat_array.reshape(self.original_shape)

    def build_huffman_tree(self, freq_dict):
        heap = [self.HuffmanNode(symbol=symbol, freq=freq) for symbol, freq in freq_dict.items()]
        heapify(heap)

        while len(heap) > 1:
            left = heappop(heap)
            right = heappop(heap)

            internal_node = self.HuffmanNode(freq=left.freq + right.freq)
            internal_node.left = left
            internal_node.right = right

            heappush(heap, internal_node)

        return heap[0]

    def build_code_table(self, node, code="", code_table=None):
        if code_table is None:
            code_table = {}

        if node.symbol is not None:
            code_table[node.symbol] = code
        if node.left is not None:
            self.build_code_table(node.left, code + "0", code_table)
        if node.right is not None:
            self.build_code_table(node.right, code + "1", code_table)

        return code_table

    def huffman_encode_ndarray(self, ndarray):
        flat_data = self.flatten_ndarray(ndarray)
        freq_dict = defaultdict(int)
        
        for symbol in flat_data:
            freq_dict[symbol] += 1

        self.huffman_tree = self.build_huffman_tree(freq_dict)
        self.code_table = self.build_code_table(self.huffman_tree)
        self.original_shape = ndarray.shape

        encoded_data = ''.join(self.code_table[symbol] for symbol in flat_data)

        return encoded_data

    def huffman_decode_ndarray(self, encoded_data):
        current_node = self.huffman_tree

        decoded_data = []
        for bit in encoded_data:
            if bit == '0':
                current_node = current_node.left
            else:
                current_node = current_node.right

            if current_node.symbol is not None:
                decoded_data.append(current_node.symbol)
                current_node = self.huffman_tree

        flat_decoded_data = np.array(decoded_data)
        reshaped_data = self.reshape_to_ndarray(flat_decoded_data)

        return reshaped_data

'''

# Example usage:
huffman_coder = HuffmanCompressor()

# original_data = np.array([[1, 2, 3], [4, 5, 6]])
original_data = np.array([e, pi, e, pi, e, pi, e, pi], dtype=np.float64)
# Encode
encoded_data = huffman_coder.huffman_encode_ndarray(original_data)

# Decode
decoded_data = huffman_coder.huffman_decode_ndarray(encoded_data)

print("Original Data:\n", original_data, sys.getsizeof(original_data))
print("Encoded Data:\n", encoded_data, sys.getsizeof(encoded_data))
print("Decoded Data:\n", decoded_data, sys.getsizeof(decoded_data))

'''