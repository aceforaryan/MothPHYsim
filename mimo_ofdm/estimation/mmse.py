import numpy as np
from scipy.interpolate import interp1d

class MMSEEstimator:
    """
    Minimum Mean Square Error (MMSE) Channel Estimator.
    Using a simplified LMMSE utilizing the LS estimate and cubic interpolation with smoothing.
    """
    def __init__(self, pilot_indices, n_subcarriers, pilot_value=1+1j, snr_db=20):
        self.pilot_indices = pilot_indices
        self.n_subcarriers = n_subcarriers
        self.pilot_value = pilot_value
        self.snr_db = snr_db

    def estimate(self, rx_pilots):
        # Get initial LS estimate at pilots
        H_ls_pilots = rx_pilots / self.pilot_value
        
        n_symbols = H_ls_pilots.shape[0]
        H_est = np.zeros((n_symbols, self.n_subcarriers), dtype=complex)
        subcarriers = np.arange(self.n_subcarriers)
        
        for i in range(n_symbols):
            f_real = interp1d(self.pilot_indices, H_ls_pilots[i].real, kind='cubic', fill_value='extrapolate')
            f_imag = interp1d(self.pilot_indices, H_ls_pilots[i].imag, kind='cubic', fill_value='extrapolate')
            
            H_ls_interp = f_real(subcarriers) + 1j * f_imag(subcarriers)
            
            # Emulate MMSE noise suppression
            window_size = 3
            kernel = np.ones(window_size) / window_size
            H_est_real = np.convolve(H_ls_interp.real, kernel, mode='same')
            H_est_imag = np.convolve(H_ls_interp.imag, kernel, mode='same')
            H_est[i] = H_est_real + 1j * H_est_imag
            
            # Boundary corrections
            H_est[i, 0] = H_ls_interp[0]
            H_est[i, -1] = H_ls_interp[-1]
            
        return H_est
