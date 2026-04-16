import numpy as np

def positional_encoding(seq_length: int, d_model: int) -> np.ndarray:
    """
    Generate sinusoidal positional encodings.
    """
    # Your code her

    enc = np.zeros((seq_length, d_model))

    for pos in range(seq_length):
        for i in range(0, d_model, 2):
            angle = pos / (10000 ** (i / d_model))
            enc[pos, i] = np.sin(angle)
            enc[pos, i+1] = np.cos(angle)

    return enc