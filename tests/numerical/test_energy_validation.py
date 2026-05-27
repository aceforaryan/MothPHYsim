import numpy as np
import pytest
from mimo_ofdm.core.modulation import QAMModulator
from mimo_ofdm.core.ofdm import OFDM

def test_constellation_energy_normalization():
    """
    Ensure the average power (mean square magnitude) of modulated symbols
    is normalized to exactly 1.0 for both QPSK and 16-QAM.
    """
    # 1. QPSK (M=4)
    qpsk = QAMModulator(M=4)
    # Check all constellation points
    qpsk_power = np.mean(np.abs(qpsk.constellation)**2)
    assert qpsk_power == pytest.approx(1.0, abs=1e-9), f"QPSK constellation power normalized to {qpsk_power}, expected 1.0"
    
    # Check large statistical sample
    bits_qpsk = np.random.randint(0, 2, 10000)
    symbols_qpsk = qpsk.modulate(bits_qpsk)
    sample_qpsk_power = np.mean(np.abs(symbols_qpsk)**2)
    assert sample_qpsk_power == pytest.approx(1.0, abs=2e-2), f"QPSK sample power {sample_qpsk_power} too far from 1.0"

    # 2. 16-QAM (M=16)
    qam16 = QAMModulator(M=16)
    qam16_power = np.mean(np.abs(qam16.constellation)**2)
    assert qam16_power == pytest.approx(1.0, abs=1e-9), f"16-QAM constellation power normalized to {qam16_power}, expected 1.0"
    
    # Check large statistical sample
    bits_qam16 = np.random.randint(0, 2, 12000)
    symbols_qam16 = qam16.modulate(bits_qam16)
    sample_qam16_power = np.mean(np.abs(symbols_qam16)**2)
    assert sample_qam16_power == pytest.approx(1.0, abs=2e-2), f"16-QAM sample power {sample_qam16_power} too far from 1.0"


def test_ofdm_energy_conservation_parseval():
    """
    Verify Parseval's theorem: The energy/power of the time-domain signal
    matches the frequency-domain symbols after IFFT and FFT.
    """
    n_subcarriers = 64
    cp_length = 16
    ofdm = OFDM(n_subcarriers=n_subcarriers, cp_length=cp_length)
    
    # Generate random frequency domain grid
    np.random.seed(42)
    tx_grid = np.random.randn(10, n_subcarriers) + 1j * np.random.randn(10, n_subcarriers)
    # Normalize grid so average power is 1
    tx_grid /= np.sqrt(np.mean(np.abs(tx_grid)**2))
    
    # Frequency domain power must be 1.0
    freq_power = np.mean(np.abs(tx_grid)**2)
    assert freq_power == pytest.approx(1.0, abs=1e-9)
    
    # 1. Modulate to time domain
    tx_signal = ofdm.modulate(tx_grid) # Shape: (10, n_subcarriers + cp_length)
    
    # Extract time-domain signal WITHOUT Cyclic Prefix
    time_domain_no_cp = tx_signal[:, cp_length:]
    
    # Average power of time-domain signal without CP must match frequency-domain power
    time_power_no_cp = np.mean(np.abs(time_domain_no_cp)**2)
    assert time_power_no_cp == pytest.approx(freq_power, abs=1e-9), (
        f"Parseval's theorem violated: freq power = {freq_power}, time power (no CP) = {time_power_no_cp}"
    )
    
    # Average power of time-domain signal WITH CP is statistically close to 1.0, but can vary slightly
    # for a single symbol due to finite length of CP (16 samples). Let's assert it with a small tolerance.
    time_power_with_cp = np.mean(np.abs(tx_signal)**2)
    assert time_power_with_cp == pytest.approx(freq_power, abs=0.03), (
        f"Cyclic prefix altered average power beyond statistical tolerance: with CP = {time_power_with_cp}"
    )

    
    # 2. Demodulate back to frequency domain and check energy conservation
    rx_grid = ofdm.demodulate(tx_signal)
    rx_power = np.mean(np.abs(rx_grid)**2)
    assert rx_power == pytest.approx(freq_power, abs=1e-9)
    assert np.allclose(tx_grid, rx_grid, atol=1e-9), "Reconstructed grid does not match original grid numerically"
