import numpy as np

class AdaptiveEstimator:
    """
    Tracks fading channel across OFDM symbols using an EWMA (Exponentially Weighted Moving Average).
    """
    def __init__(self, base_estimator, alpha=0.9):
        """
        Args:
            base_estimator: An instance of LSEstimator or MMSEEstimator
            alpha: forgetting factor (1 = trust current estimate, 0 = fully trust previous)
        """
        self.base_estimator = base_estimator
        self.alpha = alpha
        self.H_prev = None

    def estimate(self, rx_pilots):
        H_curr_est = self.base_estimator.estimate(rx_pilots)
        
        if self.H_prev is None:
            self.H_prev = H_curr_est
            return H_curr_est
            
        H_tracked = np.zeros_like(H_curr_est)
        for i in range(H_curr_est.shape[0]):
            H_tracked[i] = self.alpha * H_curr_est[i] + (1 - self.alpha) * self.H_prev[-1]
            self.H_prev = np.append(self.H_prev, [H_tracked[i]], axis=0)
            
        # Keep only the last state to save memory
        self.H_prev = self.H_prev[-1:]
        
        return H_tracked
