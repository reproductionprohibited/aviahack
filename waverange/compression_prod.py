import numpy as np

def quantize_coefficient(a, Zk, Qk):
    if a > Zk / 2:
        p = np.floor((a - Zk / 2) / Qk) + 1
    elif -Zk / 2 <= a <= Zk / 2:
        p = 0
    else:
        p = np.floor((a + Zk / 2) / Qk) - 1

    return int(p)

# Vectorize the quantization function for ndarray
vectorized_quantize = np.vectorize(quantize_coefficient)

def quantize_coefficients(coefficients, Zk, Qk):
    return vectorized_quantize(coefficients, Zk, Qk)

# Example usage:
# Assuming 'coefficients' is your 2D ndarray of wavelet coefficients
coefficients = np.random.rand(3, 3)

# Set quantization parameters
Zk = 0.5
Qk = 0.1

# Quantize the coefficients
quantized_coefficients = quantize_coefficients(coefficients, Zk, Qk)

# Print the original and quantized coefficients
print("Original Coefficients:")
print(coefficients)
print("\nQuantized Coefficients:")
print(quantized_coefficients)