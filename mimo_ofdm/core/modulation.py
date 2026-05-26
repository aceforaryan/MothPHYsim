import numpy as np

class QAMModulator:
    """
    QAM/QPSK Modulator and Demodulator.
    For QPSK, M=4. For 16-QAM, M=16.
    """
    def __init__(self, M=4):
        self.M = M
        self.k = int(np.log2(self.M)) # Bits per symbol
        if self.M == 4:
            self.constellation = np.array([
                1+1j, -1+1j, -1-1j, 1-1j
            ]) / np.sqrt(2)
        elif self.M == 16:
            # 16-QAM mapping
            mapping = np.array([
                -3+3j, -1+3j, 1+3j, 3+3j,
                -3+1j, -1+1j, 1+1j, 3+1j,
                -3-1j, -1-1j, 1-1j, 3-1j,
                -3-3j, -1-3j, 1-3j, 3-3j
            ])
            self.constellation = mapping / np.sqrt(10) # Normalize average power to 1
        else:
            raise ValueError("Only QPSK (M=4) and 16-QAM (M=16) are currently supported.")

    def modulate(self, bits):
        """
        Maps an array of bits (0s and 1s) to complex constellation symbols.
        Padding is added if the number of bits is not a multiple of k.
        """
        bits = np.asarray(bits, dtype=int)
        
        # Pad with zeros if necessary
        pad_len = (self.k - len(bits) % self.k) % self.k
        if pad_len > 0:
            bits = np.concatenate([bits, np.zeros(pad_len, dtype=int)])
            
        reshaped_bits = bits.reshape(-1, self.k)
        
        symbols = np.zeros(len(reshaped_bits), dtype=complex)
        for i, b in enumerate(reshaped_bits):
            idx = int("".join(str(x) for x in b), 2)
            symbols[i] = self.constellation[idx]
            
        return symbols

    def demodulate(self, symbols):
        """
        Maps complex received symbols back to bits using minimum distance decoding.
        """
        symbols = np.asarray(symbols)
        # Using broadcasting to find minimum distance to all constellation points
        # symbols shape: (N,)
        # constellation shape: (M,)
        distances = np.abs(symbols[:, np.newaxis] - self.constellation[np.newaxis, :])
        indices = np.argmin(distances, axis=1)
        
        bits = []
        for idx in indices:
            b = format(idx, f'0{self.k}b')
            bits.extend([int(x) for x in b])
            
        return np.array(bits)
