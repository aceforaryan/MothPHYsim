import numpy as np

class MMSEEqualizer:
    """
    MMSE Equalizer.
    Takes into account the noise variance to prevent noise amplification
    at deep fades.
    """
    def __init__(self, snr_db):
        self.snr_db = snr_db
        self.noise_variance = 10 ** (-self.snr_db / 10)

    def equalize(self, rx_data, H_est, data_indices):
        """
        Args:
            rx_data: Received data symbols (n_symbols, n_data_subcarriers)
            H_est: Estimated channel across all subcarriers (n_symbols, n_subcarriers)
            data_indices: Indices of data subcarriers
        Returns:
            tx_est: Equalized data symbols
        """
        H_data = H_est[:, data_indices]
        
        # MMSE formula: W = H^* / (|H|^2 + N0)
        W = np.conj(H_data) / (np.abs(H_data)**2 + self.noise_variance)
        
        tx_est = rx_data * W
        return tx_est
