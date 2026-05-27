import numpy as np
import pytest
from mimo_ofdm.core.modulation import QAMModulator
from mimo_ofdm.core.mimo import MIMOSpatialMultiplexer
from mimo_ofdm.extensions.sphere import SphereDecoder
from mimo_ofdm.equalization.mimo_det import MIMOZFEqualizer, MIMOMMSEEqualizer

def test_sphere_decoder_noise_free():
    """
    Verify that in a noise-free MIMO channel, the Sphere Decoder perfectly
    recovers the transmitted constellation symbols.
    """
    np.random.seed(100)
    n_tx = 2
    n_rx = 2
    
    # Use QPSK constellation
    modulator = QAMModulator(M=4)
    constellation = modulator.constellation
    decoder = SphereDecoder(constellation)
    
    # Generate random transmit symbols
    tx_symbols = np.random.choice(constellation, n_tx)
    
    # Generate channel H (n_rx, n_tx)
    multiplexer = MIMOSpatialMultiplexer(n_tx=n_tx, n_rx=n_rx)
    H_subcarriers = multiplexer.generate_channel_matrix(n_subcarriers=1)
    H = H_subcarriers[:, :, 0] # (n_rx, n_tx)
    
    # Apply channel y = H*x
    y = np.dot(H, tx_symbols)
    
    # Decode
    rx_symbols = decoder.decode(y, H)
    
    # Check perfect recovery
    assert np.allclose(tx_symbols, rx_symbols), (
        f"Sphere Decoder failed noise-free recovery. Transmitted: {tx_symbols}, Received: {rx_symbols}"
    )

def test_sphere_decoder_ml_optimality():
    """
    Verify the Maximum Likelihood optimality of the Sphere Decoder.
    For any noisy received vector y, the decoded vector x_sd must satisfy:
        ||y - H * x_sd||^2 <= ||y - H * x_other||^2
    where x_other is the symbol vector found by ZF or MMSE.
    """
    np.random.seed(200)
    n_tx = 3
    n_rx = 3
    
    # Use 16-QAM constellation
    modulator = QAMModulator(M=16)
    constellation = modulator.constellation
    decoder = SphereDecoder(constellation)
    
    # Generate random transmit symbols
    tx_symbols = np.random.choice(constellation, n_tx)
    
    # Generate channel H
    multiplexer = MIMOSpatialMultiplexer(n_tx=n_tx, n_rx=n_rx)
    H_subcarriers = multiplexer.generate_channel_matrix(n_subcarriers=1)
    H = H_subcarriers[:, :, 0]
    
    # Add significant noise
    y_clean = np.dot(H, tx_symbols)
    noise = 0.8 * (np.random.randn(n_rx) + 1j * np.random.randn(n_rx))
    y = y_clean + noise
    
    # 1. Sphere Decode
    rx_symbols_sd = decoder.decode(y, H)
    dist_sd = np.sum(np.abs(y - np.dot(H, rx_symbols_sd))**2)
    
    # 2. ZF Decode (with slicing/decision to closest constellation points)
    H_inv = np.linalg.pinv(H)
    x_zf_continuous = np.dot(H_inv, y)
    # Map to closest constellation points
    x_zf_sliced = np.zeros(n_tx, dtype=complex)
    for i in range(n_tx):
        dists = np.abs(x_zf_continuous[i] - constellation)
        x_zf_sliced[i] = constellation[np.argmin(dists)]
        
    dist_zf = np.sum(np.abs(y - np.dot(H, x_zf_sliced))**2)
    
    # 3. MMSE Decode
    snr_db = 10
    noise_var = 10**(-snr_db/10)
    W_mmse = np.linalg.inv(H.conj().T @ H + noise_var * np.eye(n_tx)) @ H.conj().T
    x_mmse_continuous = np.dot(W_mmse, y)
    x_mmse_sliced = np.zeros(n_tx, dtype=complex)
    for i in range(n_tx):
        dists = np.abs(x_mmse_continuous[i] - constellation)
        x_mmse_sliced[i] = constellation[np.argmin(dists)]
        
    dist_mmse = np.sum(np.abs(y - np.dot(H, x_mmse_sliced))**2)
    
    # Print comparison
    print(f"Distances: SD={dist_sd:.4f}, ZF={dist_zf:.4f}, MMSE={dist_mmse:.4f}")
    
    # The sphere decoder must find a solution with distance LESS THAN OR EQUAL TO the other detectors
    assert dist_sd <= dist_zf + 1e-12, f"Sphere Decoder was sub-optimal compared to ZF! SD={dist_sd}, ZF={dist_zf}"
    assert dist_sd <= dist_mmse + 1e-12, f"Sphere Decoder was sub-optimal compared to MMSE! SD={dist_sd}, MMSE={dist_mmse}"
