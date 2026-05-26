import numpy as np
from mimo_ofdm.core.modulation import QAMModulator

def test_qpsk_mod_demod():
    mod = QAMModulator(M=4)
    bits = np.array([0, 1, 1, 0, 1, 1])
    symbols = mod.modulate(bits)
    demod_bits = mod.demodulate(symbols)
    assert np.array_equal(bits, demod_bits[:len(bits)])

def test_16qam_mod_demod():
    mod = QAMModulator(M=16)
    bits = np.random.randint(0, 2, 100)
    symbols = mod.modulate(bits)
    demod_bits = mod.demodulate(symbols)
    assert np.array_equal(bits, demod_bits[:len(bits)])
