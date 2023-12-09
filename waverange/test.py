import pywt
import numpy as np
import zlib

def wavelet_transform(data):
    # Apply 1D CDF 9/7 wavelet transform along each axis
    coeffs = [pywt.dwt(data[:, i], 'bior2.2', mode='periodization') for i in range(data.shape[1])]

    return coeffs

def quantization(coeffs, threshold):
    # Apply quantization to the coefficients
    quantized_coeffs = [np.round(c[0] / threshold) * threshold for c in coeffs]

    return quantized_coeffs

def entropy_encoding(quantized_coeffs):
    # Flatten and convert to bytes
    flattened_coeffs = np.concatenate([c.flatten() for c in quantized_coeffs])
    flattened_bytes = flattened_coeffs.astype(np.int16).tobytes()

    # Use zlib for entropy encoding (compression)
    compressed_data = zlib.compress(flattened_bytes, level=zlib.Z_BEST_COMPRESSION)

    return compressed_data

def compression(data, threshold):
    # Step 1: Wavelet Transform
    coeffs = wavelet_transform(data)

    # Step 2: Quantization
    quantized_coeffs = quantization(coeffs, threshold)

    # Step 3: Entropy Encoding
    compressed_data = entropy_encoding(quantized_coeffs)

    return compressed_data

def decompression(compressed_data, data_shape, threshold):
    # Decompress the data
    decompressed_data = zlib.decompress(compressed_data)

    # Convert back to array
    decompressed_coeffs = np.frombuffer(decompressed_data, dtype=np.int16)

    # Reshape to the shape of original data
    decompressed_coeffs = [c.reshape((-1, 1)) for c in np.split(decompressed_coeffs, len(decompressed_coeffs)//data_shape[0])]

    # Reverse quantization
    decompressed_coeffs = [c * threshold for c in decompressed_coeffs]

    # Inverse wavelet transform
    reconstructed_data = np.hstack([pywt.idwt(c, None, 'bior2.2', mode='periodization') for c in decompressed_coeffs])

    return reconstructed_data

# Example usage:
# Create a sample 2D ndarray
original_data = np.random.random((100, 10))

# Set the threshold value
threshold = 0.1

# Compression
compressed_data = compression(original_data, threshold)

# Decompression
reconstructed_data = decompression(compressed_data, original_data.shape, threshold)

# Check if the reconstructed data is close to the original data
is_close = np.allclose(original_data, reconstructed_data)
print(f"Are the original and reconstructed data close? {is_close}")

# If not close, print the maximum absolute difference
if not is_close:
    max_difference = np.max(np.abs(original_data - reconstructed_data))
    print(f"Maximum absolute difference: {max_difference}")

