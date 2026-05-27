import numpy as np
import pytest
from mimo_ofdm.core.modulation import QAMModulator
from mimo_ofdm.core.ofdm import OFDM
from mimo_ofdm.channel.noise import AWGNChannel
from mimo_ofdm.channel.fading import TDLChannel
from mimo_ofdm.estimation.pilots import PilotInserter
from mimo_ofdm.estimation.ls import LSEstimator
from mimo_ofdm.equalization.zf import ZeroForcingEqualizer
from mimo_ofdm.utils.metrics import calculate_ber

def test_ber_bounds():
    """
    Ensure BER is strictly bounded between 0.0 and 1.0.
    """
    tx_bits = np.array([0, 1, 0, 1, 0])
    rx_bits = np.array([0, 1, 0, 1, 0])
    assert calculate_ber(tx_bits, rx_bits) == 0.0
    
    rx_bits_all_wrong = np.array([1, 0, 1, 0, 1])
    assert calculate_ber(tx_bits, rx_bits_all_wrong) == 1.0
    
    rx_bits_partial = np.array([0, 1, 1, 1, 0])
    assert calculate_ber(tx_bits, rx_bits_partial) == 0.2

def test_high_snr_zero_ber():
    """
    In an ideal or high SNR channel (50 dB) with Least Squares estimation and ZF equalization,
    the Bit Error Rate (BER) must be exactly 0.
    """
    np.random.seed(1234)
    n_subcarriers = 64
    cp_length = 16
    n_ofdm_symbols = 10
    
    modulator = QAMModulator(M=16) # Use 16-QAM to check high constellation stability
    ofdm = OFDM(n_subcarriers=n_subcarriers, cp_length=cp_length)
    pilots = PilotInserter(n_subcarriers=n_subcarriers, pilot_spacing=4)
    channel = TDLChannel(sample_rate=1e6, delays=[0, 1e-6], powers_db=[0, -3], seed=42)
    noise = AWGNChannel(seed=42)
    estimator = LSEstimator(pilot_indices=pilots.pilot_indices, n_subcarriers=n_subcarriers)
    equalizer = ZeroForcingEqualizer()

    # Generate transmission bits
    n_bits = n_ofdm_symbols * len(pilots.data_indices) * modulator.k
    tx_bits = np.random.randint(0, 2, n_bits)
    
    # Transmitter
    tx_symbols = modulator.modulate(tx_bits).reshape(n_ofdm_symbols, -1)
    tx_grid = pilots.insert(tx_symbols)
    tx_signal = ofdm.modulate(tx_grid)
    tx_serial = tx_signal.flatten()
    
    # Channel with high SNR (50 dB)
    faded_signal, _ = channel.apply(tx_serial)
    rx_serial = noise.apply(faded_signal, snr_db=50)
    
    # Receiver
    rx_signal = rx_serial.reshape(n_ofdm_symbols, -1)
    rx_grid = ofdm.demodulate(rx_signal)
    rx_pilots, rx_data = pilots.extract(rx_grid)
    
    H_est = estimator.estimate(rx_pilots)
    eq_data = equalizer.equalize(rx_data, H_est, pilots.data_indices)
    rx_bits = modulator.demodulate(eq_data.flatten())
    
    ber = calculate_ber(tx_bits, rx_bits)
    assert ber == 0.0, f"High SNR BER was {ber}, expected exactly 0.0"

def test_snr_monotonicity():
    """
    Rigorously verify that statistically, as SNR increases, the BER decreases or remains flat.
    If BER increases significantly at a higher SNR, it indicates scaling or noise-generation bugs.
    """
    np.random.seed(42)
    n_subcarriers = 64
    cp_length = 16
    n_ofdm_symbols = 40 # Large enough symbol size to suppress statistical variance
    
    modulator = QAMModulator(M=4) # QPSK is more sensitive to monotonic trend across SNR
    ofdm = OFDM(n_subcarriers=n_subcarriers, cp_length=cp_length)
    pilots = PilotInserter(n_subcarriers=n_subcarriers, pilot_spacing=4)
    channel = TDLChannel(sample_rate=1e6, delays=[0, 1e-6, 3e-6], powers_db=[0, -3, -10], seed=42)
    noise = AWGNChannel(seed=42)
    estimator = LSEstimator(pilot_indices=pilots.pilot_indices, n_subcarriers=n_subcarriers)
    equalizer = ZeroForcingEqualizer()

    n_bits = n_ofdm_symbols * len(pilots.data_indices) * modulator.k
    tx_bits = np.random.randint(0, 2, n_bits)
    
    tx_symbols = modulator.modulate(tx_bits).reshape(n_ofdm_symbols, -1)
    tx_grid = pilots.insert(tx_symbols)
    tx_signal = ofdm.modulate(tx_grid)
    tx_serial = tx_signal.flatten()
    faded_signal, _ = channel.apply(tx_serial)
    
    snr_levels = [0, 5, 10, 15, 20, 25]
    bers = []
    
    for snr in snr_levels:
        rx_serial = noise.apply(faded_signal, snr_db=snr)
        rx_signal = rx_serial.reshape(n_ofdm_symbols, -1)
        rx_grid = ofdm.demodulate(rx_signal)
        rx_pilots, rx_data = pilots.extract(rx_grid)
        
        H_est = estimator.estimate(rx_pilots)
        eq_data = equalizer.equalize(rx_data, H_est, pilots.data_indices)
        rx_bits = modulator.demodulate(eq_data.flatten())
        
        ber = calculate_ber(tx_bits, rx_bits)
        bers.append(ber)
        
    # Print the observed BERs for debugging
    print(f"Observed BERs: {dict(zip(snr_levels, bers))}")
    
    # Assert monotonic decrease (each BER must be less than or equal to the previous,
    # allowing for a tiny statistical tolerance of 0.015 to prevent random test failures)
    tolerance = 0.015
    for i in range(1, len(bers)):
        assert bers[i] <= bers[i-1] + tolerance, (
            f"Non-monotonic BER detected: BER at {snr_levels[i]}dB ({bers[i]}) was larger than at "
            f"{snr_levels[i-1]}dB ({bers[i-1]})"
        )
        
    # Check that high SNR has a significantly lower BER than low SNR
    assert bers[-1] < bers[0] * 0.1, f"High SNR BER ({bers[-1]}) did not decrease significantly compared to low SNR ({bers[0]})"
