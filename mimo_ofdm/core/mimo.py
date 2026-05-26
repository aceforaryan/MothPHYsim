import numpy as np

class MIMOSpatialMultiplexer:
    """
    Handles Spatial Multiplexing for MIMO Systems.
    Default configuration is 2x2.
    """
    def __init__(self, n_tx=2, n_rx=2):
        self.n_tx = n_tx
        self.n_rx = n_rx

    def multiplex(self, symbols):
        """
        Distributes data symbols across multiple transmit antennas.
        
        Args:
            symbols: Flat array of complex symbols.
        Returns:
            tx_streams: Array of shape (n_tx, N_symbols_per_antenna)
        """
        symbols = np.asarray(symbols)
        # Pad if necessary so it's perfectly divisible by n_tx
        pad_len = (self.n_tx - len(symbols) % self.n_tx) % self.n_tx
        if pad_len > 0:
            symbols = np.concatenate([symbols, np.zeros(pad_len, dtype=complex)])
            
        # Reshape: each row is a stream for one antenna
        tx_streams = symbols.reshape(-1, self.n_tx).T
        return tx_streams
        
    def generate_channel_matrix(self, n_subcarriers):
        """
        Generates a flat Rayleigh fading channel matrix for testing/stubbing.
        
        Returns:
            H: Channel matrix of shape (n_rx, n_tx, n_subcarriers)
        """
        H = np.sqrt(0.5) * (np.random.randn(self.n_rx, self.n_tx, n_subcarriers) + 
                            1j * np.random.randn(self.n_rx, self.n_tx, n_subcarriers))
        return H
