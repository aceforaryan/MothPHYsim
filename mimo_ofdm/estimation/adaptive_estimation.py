import numpy as np


class AdaptiveEstimator:
    """
    Tracks a time-varying fading channel across OFDM symbols using an
    Exponentially Weighted Moving Average (EWMA).

    The EWMA update rule applied per-symbol is:
        H_tracked[i] = alpha * H_current[i] + (1 - alpha) * H_tracked[i-1]

    where alpha ∈ (0, 1] is the forgetting factor:
        - alpha → 1  : trust the instantaneous LS estimate (fast tracking, more noise)
        - alpha → 0  : trust the historical average heavily (slow tracking, noise-smoothed)
    """

    def __init__(self, base_estimator, alpha=0.9):
        """
        Args:
            base_estimator: An instance of LSEstimator or MMSEEstimator.
            alpha: Forgetting factor in (0, 1].  1 = no smoothing (pure LS),
                   0 = never update (pathological).  Default 0.9.
        """
        self.base_estimator = base_estimator
        self.alpha = alpha
        self.H_prev = None  # Shape: (n_subcarriers,) — last tracked symbol

    def estimate(self, rx_pilots):
        """
        Estimates the channel frequency response with EWMA temporal smoothing.

        Args:
            rx_pilots: Received pilot symbols, shape (n_symbols, n_pilots).
        Returns:
            H_tracked: Smoothed channel estimate, shape (n_symbols, n_subcarriers).
        """
        # Get instantaneous LS/MMSE estimate for the entire batch
        H_curr_est = self.base_estimator.estimate(rx_pilots)  # (n_symbols, n_subcarriers)

        n_symbols = H_curr_est.shape[0]
        H_tracked = np.zeros_like(H_curr_est)

        for i in range(n_symbols):
            if self.H_prev is None:
                # Cold-start: initialise state from first symbol's LS estimate
                H_tracked[i] = H_curr_est[i]
            else:
                # Causal EWMA: blend current estimate with *previous tracked* symbol
                H_tracked[i] = self.alpha * H_curr_est[i] + (1.0 - self.alpha) * self.H_prev

            # Update state — always use the tracked (smoothed) value as the prior
            self.H_prev = H_tracked[i]

        return H_tracked

    def reset(self):
        """Resets the internal state so the estimator can be reused for a new frame."""
        self.H_prev = None
