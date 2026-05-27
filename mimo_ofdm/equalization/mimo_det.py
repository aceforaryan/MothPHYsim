import numpy as np

class MIMOZFEqualizer:
    """
    Zero-Forcing (ZF) MIMO Equalizer/Detector.
    Separates spatially multiplexed streams by inverting the channel matrix H at each subcarrier.
    """
    def equalize(self, rx_grid, H):
        """
        Args:
            rx_grid: Received grid of shape (n_rx, n_symbols, n_subcarriers)
            H: Channel matrix of shape (n_rx, n_tx, n_subcarriers)
        Returns:
            tx_est: Estimated transmit symbols of shape (n_tx, n_symbols, n_subcarriers)
        """
        n_rx, n_symbols, n_subcarriers = rx_grid.shape
        _, n_tx, _ = H.shape
        tx_est = np.zeros((n_tx, n_symbols, n_subcarriers), dtype=complex)
        
        for k in range(n_subcarriers):
            H_k = H[:, :, k]
            # Compute pseudo-inverse of H_k
            H_inv = np.linalg.pinv(H_k)
            # H_inv: (n_tx, n_rx), rx_grid[:, :, k]: (n_rx, n_symbols)
            tx_est[:, :, k] = H_inv @ rx_grid[:, :, k]
            
        return tx_est


class MIMOMMSEEqualizer:
    """
    Minimum Mean Squared Error (MMSE) MIMO Equalizer/Detector.
    Accounts for noise variance to prevent excessive noise amplification, especially in deep fades.
    """
    def __init__(self, snr_db):
        """
        Args:
            snr_db: Signal-to-Noise Ratio in dB.
        """
        self.snr_db = snr_db
        self.noise_variance = 10 ** (-snr_db / 10)

    def equalize(self, rx_grid, H):
        """
        Args:
            rx_grid: Received grid of shape (n_rx, n_symbols, n_subcarriers)
            H: Channel matrix of shape (n_rx, n_tx, n_subcarriers)
        Returns:
            tx_est: Estimated transmit symbols of shape (n_tx, n_symbols, n_subcarriers)
        """
        n_rx, n_symbols, n_subcarriers = rx_grid.shape
        _, n_tx, _ = H.shape
        tx_est = np.zeros((n_tx, n_symbols, n_subcarriers), dtype=complex)
        
        for k in range(n_subcarriers):
            H_k = H[:, :, k] # (n_rx, n_tx)
            H_k_H = np.conj(H_k).T # (n_tx, n_rx)
            # MMSE filter matrix: W = (H^H * H + sigma^2 * I)^-1 * H^H
            W = np.linalg.inv(H_k_H @ H_k + self.noise_variance * np.eye(n_tx)) @ H_k_H
            # W: (n_tx, n_rx), rx_grid[:, :, k]: (n_rx, n_symbols)
            tx_est[:, :, k] = W @ rx_grid[:, :, k]
            
        return tx_est
