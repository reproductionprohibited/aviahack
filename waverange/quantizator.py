import numpy as np
from math import log2


class Quantizator:
    def __init__(self):
        self.data_min = None
        self.step_size = None

    def uniform_quantization(array: np.ndarray) -> np.ndarray:
        min_val = np.min(array)
        max_val = np.max(array)
        num_bits = 2
        temp = 2 ** int(log2(max_val - min_val) - 3)
        while (num_bits < (max_val - min_val) / temp):
            num_bits = num_bits << 1
        # Вычисление шага квантования
        q_step = (max_val - min_val) / (2**num_bits)
        
        # Применение квантования к массиву
        quantized_array = np.round((array - min_val) / q_step)
        
        return [quantized_array, q_step, min_val]

    def dequantize(self, quantized: np.ndarray) -> np.ndarray:

        array = np.random.rand(5, 5, 5) * 1000

        quantized_array, step, val = self.uniform_quantization(array)

        unquantized_array = quantized_array * step + val

        # Вывод исходного и квантованного массива
        print(step, val, np.max(quantized_array))
        print(np.mean((array - unquantized_array) ** 2))

    def mse(self, data: np.ndarray, dequantized: np.ndarray) -> np.float64:
        print(np.mean((data - dequantized   ) ** 2))