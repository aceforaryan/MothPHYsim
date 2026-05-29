import numpy as np
from scipy.interpolate import interp1d


class MMSEEstimator:
    """
    Linear MMSE (LMMSE) Channel Estimator with diagonal Wiener shrinkage.

    The estimator applies a per-subcarrier scalar LMMSE filter to the initial
    LS estimates at pilot positions before interpolating across all subcarriers.

    For an AWGN-corrupted pilot observation  Y_p = H_p · X_p + N_p  the optimal
    diagonal (per-pilot) LMMSE weight is:

        W = SNR / (SNR + 1)          [assuming E[|H|²] = 1, E[|N|²] = σ²]

    This shrinks the LS estimate toward zero by the factor W, minimising the
    mean-square error.  At high SNR, W → 1 and LMMSE collapses to LS; at low
    SNR, W → 0 and the estimator hedges by pulling toward the prior mean (zero).

    Interpolation uses cubic splines, which are appropriate here because the
    LMMSE-shrunk pilot values are already denoised, removing the overshoot risk
    that cubic interpolation has on raw (noisy) LS pilot estimates.
    """

    def __init__(self, pilot_indices, n_subcarriers, pilot_value=1 + 1j, snr_db=20):
        """
        Args:
            pilot_indices:  Array of pilot subcarrier positions.
            n_subcarriers:  Total number of OFDM subcarriers.
            pilot_value:    Known pilot symbol value (default 1+1j).
            snr_db:         Operating SNR in dB — used to compute the Wiener weight.
        """
        self.pilot_indices = np.asarray(pilot_indices)
        self.n_subcarriers = n_subcarriers
        self.pilot_value = pilot_value
        self.snr_db = snr_db

        # Pre-compute the scalar Wiener shrinkage weight from SNR
        snr_linear = 10 ** (snr_db / 10.0)
        self.wiener_weight = snr_linear / (snr_linear + 1.0)

    def estimate(self, rx_pilots):
        """
        Estimates the channel frequency response using diagonal LMMSE + interpolation.

        Args:
            rx_pilots: Received pilot symbols, shape (n_symbols, n_pilots).
        Returns:
            H_est: Estimated channel across all subcarriers, shape (n_symbols, n_subcarriers).
        """
        # Step 1 — LS estimate at pilot subcarriers: H_LS = Y_p / X_p
        H_ls_pilots = rx_pilots / self.pilot_value  # (n_symbols, n_pilots)

        # Step 2 — Apply scalar Wiener shrinkage (diagonal LMMSE)
        #   This minimises MSE by trading a small bias for a large variance reduction,
        #   especially beneficial at low-to-moderate SNR.
        H_lmmse_pilots = self.wiener_weight * H_ls_pilots  # (n_symbols, n_pilots)

        n_symbols = H_lmmse_pilots.shape[0]
        H_est = np.zeros((n_symbols, self.n_subcarriers), dtype=complex)
        subcarriers = np.arange(self.n_subcarriers)

        # Step 3 — Interpolate the denoised pilot estimates across all subcarriers.
        #   Cubic interpolation is appropriate here because the inputs (H_lmmse_pilots)
        #   have already been denoised, avoiding the overshoot risk of cubic on raw LS.
        for i in range(n_symbols):
            f_real = interp1d(
                self.pilot_indices,
                H_lmmse_pilots[i].real,
                kind="cubic",
                fill_value="extrapolate",
            )
            f_imag = interp1d(
                self.pilot_indices,
                H_lmmse_pilots[i].imag,
                kind="cubic",
                fill_value="extrapolate",
            )
            H_est[i] = f_real(subcarriers) + 1j * f_imag(subcarriers)

        return H_est
