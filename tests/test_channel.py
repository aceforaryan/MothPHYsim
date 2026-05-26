import numpy as np
from mimo_ofdm.channel.noise import AWGNChannel

def test_awgn_channel():
    noise = AWGNChannel(seed=42)
    signal = np.ones(100)
    
    # At very high SNR, signal should be mostly unchanged
    noisy = noise.apply(signal, snr_db=100)
    assert np.allclose(signal, noisy, atol=1e-2)
