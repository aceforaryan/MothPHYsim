import numpy as np

class ZeroForcingEqualizer:
    """
    Zero-Forcing (ZF) Equalizer.
    Inverts the estimated channel.
    """
    def equalize(self, rx_data, H_est, data_indices):
        """
        Args:
            rx_data: Received data symbols (n_symbols, n_data_subcarriers)
            H_est: Estimated channel across all subcarriers (n_symbols, n_subcarriers)
            data_indices: Indices of data subcarriers
        Returns:
            tx_est: Equalized data symbols
        """
        # Extract channel estimates at data subcarriers
        H_data = H_est[:, data_indices]
        
        epsilon = 1e-10
        H_inv = 1.0 / (H_data + epsilon)
        
        tx_est = rx_data * H_inv
        return tx_est
