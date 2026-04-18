import numpy as np

def pad_sequences(seqs, pad_value=0, max_len=None):
    """
    Returns: np.ndarray of shape (N, L) where:
      N = len(seqs)
      L = max_len if provided else max(len(seq) for seq in seqs) or 0
    """

    if max_len is None:
        max_len = max(len(seq) for seq in seqs)

    for i, seq in enumerate(seqs):
        seqs[i] = seq[:max_len]

        if len(seqs[i]) < max_len:
            seqs[i].extend((max_len - len(seq))*[pad_value])
            
    return np.asarray(seqs)