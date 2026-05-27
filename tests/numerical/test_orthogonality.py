import numpy as np
import pytest
from mimo_ofdm.core.ofdm import OFDM

def test_ofdm_subcarrier_orthogonality():
    """
    Verify that subcarriers are strictly orthogonal in an ideal channel.
    If subcarrier k is active, all other subcarriers must be exactly 0 (or numerical noise)
    after modulation and demodulation.
    """
    n_subcarriers = 64
    cp_length = 16
    ofdm = OFDM(n_subcarriers=n_subcarriers, cp_length=cp_length)
    
    for k in range(n_subcarriers):
        # Create a single active subcarrier symbol
        tx_symbols = np.zeros((1, n_subcarriers), dtype=complex)
        tx_symbols[0, k] = 1.0 + 0j
        
        # Modulate to time domain
        tx_signal = ofdm.modulate(tx_symbols)
        
        # Demodulate back to frequency domain (ideal channel)
        rx_symbols = ofdm.demodulate(tx_signal)
        
        # Check active subcarrier
        assert np.abs(rx_symbols[0, k] - 1.0) < 1e-14, f"Active subcarrier {k} value was {rx_symbols[0, k]}, expected 1.0"
        
        # Check all other inactive subcarriers
        inactive_indices = [i for i in range(n_subcarriers) if i != k]
        for idx in inactive_indices:
            assert np.abs(rx_symbols[0, idx]) < 1e-14, (
                f"Leakage from subcarrier {k} to subcarrier {idx}: value = {rx_symbols[0, idx]}"
            )

def test_noise_leakage_and_independence():
    """
    Verify that white Gaussian noise applied to the time-domain signal
    remains white and identically distributed in the frequency domain,
    without creating systematic correlations (leakage) between subcarriers.
    """
    n_subcarriers = 64
    cp_length = 16
    ofdm = OFDM(n_subcarriers=n_subcarriers, cp_length=cp_length)
    
    np.random.seed(42)
    n_runs = 2000
    
    # Transmit all-zeros to isolate noise
    tx_symbols = np.zeros((n_runs, n_subcarriers), dtype=complex)
    tx_signal = ofdm.modulate(tx_symbols)
    
    # Add time-domain AWGN
    noise_variance = 0.5
    noise = np.sqrt(noise_variance / 2) * (np.random.randn(*tx_signal.shape) + 1j * np.random.randn(*tx_signal.shape))
    rx_signal = tx_signal + noise
    
    # Demodulate to frequency domain
    rx_symbols = ofdm.demodulate(rx_signal)
    
    # Check that frequency domain noise variance matches time domain noise variance
    freq_noise_variance = np.var(rx_symbols)
    assert freq_noise_variance == pytest.approx(noise_variance, rel=5e-2), (
        f"Frequency domain noise variance {freq_noise_variance} differs from time domain {noise_variance}"
    )
    
    # Check covariance between different subcarriers to verify independence (no systematic leakage)
    cov_matrix = np.cov(rx_symbols, rowvar=False)
    diagonal_elements = np.diagonal(cov_matrix)
    
    # Average off-diagonal absolute value should be very small (close to 0, indicating independence)
    off_diagonal_mask = ~np.eye(n_subcarriers, dtype=bool)
    mean_off_diag_abs = np.mean(np.abs(cov_matrix[off_diagonal_mask]))
    
    # Check that off-diagonals are small relative to diagonal noise variance
    assert mean_off_diag_abs < 0.05 * np.mean(diagonal_elements.real), (
        f"Systematic correlation detected between subcarriers! Mean off-diagonal covariance: {mean_off_diag_abs}"
    )
