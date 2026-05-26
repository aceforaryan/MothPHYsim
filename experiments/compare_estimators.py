import numpy as np
import matplotlib.pyplot as plt
from mimo_ofdm.core.modulation import QAMModulator
from mimo_ofdm.core.ofdm import OFDM
from mimo_ofdm.channel.noise import AWGNChannel
from mimo_ofdm.channel.fading import TDLChannel
from mimo_ofdm.estimation.pilots import PilotInserter
from mimo_ofdm.estimation.ls import LSEstimator
from mimo_ofdm.estimation.mmse import MMSEEstimator
from mimo_ofdm.equalization.zf import ZeroForcingEqualizer
from mimo_ofdm.utils.metrics import calculate_ber

def run_experiment():
    snr_range = np.arange(0, 31, 5)
    ber_ls = []
    ber_mmse = []
    
    n_subcarriers = 64
    n_ofdm_symbols = 50
    
    modulator = QAMModulator(M=4) # QPSK
    ofdm = OFDM(n_subcarriers=n_subcarriers)
    pilots = PilotInserter(n_subcarriers=n_subcarriers)
    channel = TDLChannel(sample_rate=1e6, delays=[0, 1e-6], powers_db=[0, -3])
    noise = AWGNChannel()
    
    estimator_ls = LSEstimator(pilots.pilot_indices, n_subcarriers)
    equalizer = ZeroForcingEqualizer()
    
    n_bits = n_ofdm_symbols * len(pilots.data_indices) * modulator.k
    tx_bits = np.random.randint(0, 2, n_bits)
    tx_symbols = modulator.modulate(tx_bits).reshape(n_ofdm_symbols, -1)
    tx_grid = pilots.insert(tx_symbols)
    tx_serial = ofdm.modulate(tx_grid).flatten()
    
    faded_signal, _ = channel.apply(tx_serial)
    
    for snr in snr_range:
        rx_serial = noise.apply(faded_signal, snr)
        rx_grid = ofdm.demodulate(rx_serial.reshape(n_ofdm_symbols, -1))
        rx_pilots, rx_data = pilots.extract(rx_grid)
        
        # LS
        H_est_ls = estimator_ls.estimate(rx_pilots)
        eq_data_ls = equalizer.equalize(rx_data, H_est_ls, pilots.data_indices)
        rx_bits_ls = modulator.demodulate(eq_data_ls.flatten())
        ber_ls.append(calculate_ber(tx_bits, rx_bits_ls))
        
        # MMSE
        estimator_mmse = MMSEEstimator(pilots.pilot_indices, n_subcarriers, snr_db=snr)
        H_est_mmse = estimator_mmse.estimate(rx_pilots)
        eq_data_mmse = equalizer.equalize(rx_data, H_est_mmse, pilots.data_indices)
        rx_bits_mmse = modulator.demodulate(eq_data_mmse.flatten())
        ber_mmse.append(calculate_ber(tx_bits, rx_bits_mmse))
        
    plt.figure()
    plt.semilogy(snr_range, ber_ls, 'o-', label='LS Estimation')
    plt.semilogy(snr_range, ber_mmse, 's--', label='MMSE Estimation')
    plt.title("BER vs SNR: LS vs MMSE")
    plt.xlabel("SNR (dB)")
    plt.ylabel("Bit Error Rate (BER)")
    plt.grid(True, which="both", ls="-")
    plt.legend()
    plt.savefig("compare_estimators.png")
    print("Saved plot to compare_estimators.png")

if __name__ == "__main__":
    run_experiment()
