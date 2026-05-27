import numpy as np
from mimo_ofdm.core.mimo import MIMOSpatialMultiplexer
from mimo_ofdm.equalization.mimo_det import MIMOZFEqualizer, MIMOMMSEEqualizer

def test_mimo_spatial_multiplexing():
    multiplexer = MIMOSpatialMultiplexer(n_tx=2, n_rx=2)
    symbols = np.array([1, 2, 3, 4, 5, 6], dtype=complex)
    
    # 2 transmit antennas
    tx_streams = multiplexer.multiplex(symbols)
    assert tx_streams.shape == (2, 3)
    # Check reshape mapping (each row is a stream for one antenna)
    # symbols: [1, 2, 3, 4, 5, 6].reshape(-1, 2).T -> [[1, 3, 5], [2, 4, 6]]
    assert np.array_equal(tx_streams[0], [1, 3, 5])
    assert np.array_equal(tx_streams[1], [2, 4, 6])
    
    # Test padding
    odd_symbols = np.array([1, 2, 3], dtype=complex)
    tx_streams_padded = multiplexer.multiplex(odd_symbols)
    assert tx_streams_padded.shape == (2, 2)
    assert tx_streams_padded[1, 1] == 0j # Padded element

def test_mimo_zf_equalizer():
    n_tx = 2
    n_rx = 2
    n_symbols = 5
    n_subcarriers = 8
    
    # Generate random input symbols
    tx_grid = np.random.randn(n_tx, n_symbols, n_subcarriers) + 1j * np.random.randn(n_tx, n_symbols, n_subcarriers)
    
    # Generate random MIMO channel matrix (n_rx, n_tx, n_subcarriers)
    multiplexer = MIMOSpatialMultiplexer(n_tx=n_tx, n_rx=n_rx)
    H = multiplexer.generate_channel_matrix(n_subcarriers)
    
    # Apply channel to tx_grid
    rx_grid = np.zeros((n_rx, n_symbols, n_subcarriers), dtype=complex)
    for k in range(n_subcarriers):
        H_k = H[:, :, k]
        # rx_grid[:, s, k] = H_k @ tx_grid[:, s, k]
        rx_grid[:, :, k] = H_k @ tx_grid[:, :, k]
        
    equalizer = MIMOZFEqualizer()
    tx_est = equalizer.equalize(rx_grid, H)
    
    # In noise-free case, ZF should perfectly reconstruct tx_grid
    assert np.allclose(tx_grid, tx_est)

def test_mimo_mmse_equalizer():
    n_tx = 2
    n_rx = 2
    n_symbols = 5
    n_subcarriers = 8
    
    # Generate random input symbols
    tx_grid = np.random.randn(n_tx, n_symbols, n_subcarriers) + 1j * np.random.randn(n_tx, n_symbols, n_subcarriers)
    
    # Generate random MIMO channel matrix (n_rx, n_tx, n_subcarriers)
    multiplexer = MIMOSpatialMultiplexer(n_tx=n_tx, n_rx=n_rx)
    H = multiplexer.generate_channel_matrix(n_subcarriers)
    
    # Apply channel to tx_grid
    rx_grid = np.zeros((n_rx, n_symbols, n_subcarriers), dtype=complex)
    for k in range(n_subcarriers):
        H_k = H[:, :, k]
        rx_grid[:, :, k] = H_k @ tx_grid[:, :, k]
        
    # High SNR MMSE Equalizer (e.g. 100 dB) should behave almost identically to ZF
    equalizer = MIMOMMSEEqualizer(snr_db=100)
    tx_est = equalizer.equalize(rx_grid, H)
    
    assert np.allclose(tx_grid, tx_est, atol=1e-5)
