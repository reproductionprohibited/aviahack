import constriction
import numpy as np

import sys

from math import pi
e = 2.718281828459045

message = np.array([e, pi, e, pi, e, pi, e, pi], dtype=np.float64)

# Define an i.i.d. entropy model (see below for more complex models):
entropy_model = constriction.stream.model.QuantizedGaussian(-50, 50, 3.2, 9.6)

# Let's use an ANS coder in this example. See below for a Range Coder example.
encoder = constriction.stream.stack.AnsCoder()
encoder.encode_reverse(message, entropy_model)

compressed = encoder.get_compressed()
print(sys.getsizeof(message))
print(sys.getsizeof(compressed))
print(f"compressed representation: {compressed}")
print(f"(in binary: {[bin(word) for word in compressed]})")

decoder = constriction.stream.stack.AnsCoder(compressed)
decoded = decoder.decode(entropy_model, 8) # (decodes 9 symbols)
assert np.all(decoded == message)