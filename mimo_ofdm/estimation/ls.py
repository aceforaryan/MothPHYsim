import numpy as np
from scipy.interpolate import interp1d

class LSEstimator:
    """
    Least Squares (LS) Channel Estimator with interpolation.
    """
    def __init__(self, pilot_indices, n_subcarriers, pilot_value=1+1j):
        self.pilot_indices = pilot_indices
        self.n_subcarriers = n_subcarriers
        self.pilot_value = pilot_value

    def estimate(self, rx_pilots):
        """
        Estimates the channel frequency response.
        
        Args:
            rx_pilots: Array of received pilot symbols, shape (n_symbols, n_pilots)
        Returns:
            H_est: Estimated channel response across all subcarriers, shape (n_symbols, n_subcarriers)
        """
        # LS Estimation at pilot subcarriers: H_LS = Y / X
        H_ls_pilots = rx_pilots / self.pilot_value
        
        n_symbols = H_ls_pilots.shape[0]
        H_est = np.zeros((n_symbols, self.n_subcarriers), dtype=complex)
        
        subcarriers = np.arange(self.n_subcarriers)
        
        # Interpolate across subcarriers for each symbol
        for i in range(n_symbols):
            f_real = interp1d(self.pilot_indices, H_ls_pilots[i].real, kind='linear', fill_value='extrapolate')
            f_imag = interp1d(self.pilot_indices, H_ls_pilots[i].imag, kind='linear', fill_value='extrapolate')
            
            H_est[i] = f_real(subcarriers) + 1j * f_imag(subcarriers)
            
        return H_est
