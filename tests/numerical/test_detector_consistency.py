import numpy as np
import pytest
from mimo_ofdm.equalization.zf import ZeroForcingEqualizer
from mimo_ofdm.equalization.mmse_eq import MMSEEqualizer
from mimo_ofdm.equalization.mimo_det import MIMOZFEqualizer, MIMOMMSEEqualizer
from mimo_ofdm.core.mimo import MIMOSpatialMultiplexer

def test_siso_detector_consistency():
    """
    Verify that in SISO OFDM, the MMSE equalizer converges to the Zero Forcing (ZF)
    equalizer at high SNR (low noise variance).
    """
    np.random.seed(42)
    n_symbols = 10
    n_subcarriers = 64
    data_indices = np.arange(n_subcarriers)
    
    # Generate random received data and channel estimate
    rx_data = np.random.randn(n_symbols, n_subcarriers) + 1j * np.random.randn(n_symbols, n_subcarriers)
    H_est = np.random.randn(n_symbols, n_subcarriers) + 1j * np.random.randn(n_symbols, n_subcarriers)
    
    # Avoid zero channel coefficients
    H_est[np.abs(H_est) < 0.1] = 0.5
    
    # 1. Zero Forcing Equalization
    zf_equalizer = ZeroForcingEqualizer()
    zf_est = zf_equalizer.equalize(rx_data, H_est, data_indices)
    
    # 2. High-SNR MMSE Equalization (e.g. 100 dB)
    mmse_equalizer_high_snr = MMSEEqualizer(snr_db=100)
    mmse_est_high_snr = mmse_equalizer_high_snr.equalize(rx_data, H_est, data_indices)
    
    # They should be extremely close (converge to identical values)
    assert np.allclose(zf_est, mmse_est_high_snr, rtol=1e-5, atol=1e-5), (
        "SISO MMSE did not converge to ZF at 100 dB SNR"
    )

def test_mimo_detector_consistency():
    """
    Verify that in MIMO systems, the MIMOMMSEEqualizer converges to the MIMOZFEqualizer
    at high SNR (low noise variance).
    """
    np.random.seed(42)
    n_tx = 2
    n_rx = 2
    n_symbols = 5
    n_subcarriers = 8
    
    # Generate random transmitted symbols and channel matrix
    tx_grid = np.random.randn(n_tx, n_symbols, n_subcarriers) + 1j * np.random.randn(n_tx, n_symbols, n_subcarriers)
    multiplexer = MIMOSpatialMultiplexer(n_tx=n_tx, n_rx=n_rx)
    H = multiplexer.generate_channel_matrix(n_subcarriers)
    
    # Apply channel to transmitted grid (noise-free case)
    rx_grid = np.zeros((n_rx, n_symbols, n_subcarriers), dtype=complex)
    for k in range(n_subcarriers):
        rx_grid[:, :, k] = H[:, :, k] @ tx_grid[:, :, k]
        
    # 1. MIMO Zero Forcing Detection
    mimo_zf = MIMOZFEqualizer()
    zf_est = mimo_zf.equalize(rx_grid, H)
    
    # 2. High-SNR MIMO MMSE Detection (100 dB SNR)
    mimo_mmse = MIMOMMSEEqualizer(snr_db=100)
    mmse_est = mimo_mmse.equalize(rx_grid, H)
    
    # They should converge to identical estimates
    assert np.allclose(zf_est, mmse_est, rtol=1e-5, atol=1e-5), (
        "MIMO MMSE did not converge to MIMO ZF at 100 dB SNR"
    )
    
    # In noise-free case, both should perfectly reconstruct the original transmit grid
    assert np.allclose(tx_grid, zf_est, rtol=1e-9, atol=1e-9), "MIMO ZF did not perfectly reconstruct tx_grid in noise-free case"
    assert np.allclose(tx_grid, mmse_est, rtol=1e-5, atol=1e-5), "MIMO MMSE did not converge to the correct tx_grid at 100 dB SNR"
