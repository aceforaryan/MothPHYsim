import numpy as np
import matplotlib.pyplot as plt
from mimo_ofdm.core.modulation import QAMModulator
from mimo_ofdm.core.ofdm import OFDM
from mimo_ofdm.channel.noise import AWGNChannel
from mimo_ofdm.channel.fading import TDLChannel
from mimo_ofdm.estimation.pilots import PilotInserter
from mimo_ofdm.estimation.ls import LSEstimator
from mimo_ofdm.equalization.zf import ZeroForcingEqualizer
from mimo_ofdm.utils.metrics import calculate_ber

def run_experiment():
    snr_range = np.arange(0, 31, 5)
    ber_qpsk = []
    ber_16qam = []

    n_subcarriers = 64
    n_ofdm_symbols = 200  # Increased from 50 to ensure enough bits at high SNR
    
    ofdm = OFDM(n_subcarriers=n_subcarriers)
    pilots = PilotInserter(n_subcarriers=n_subcarriers)
    channel = TDLChannel(sample_rate=1e6, delays=[0, 1e-6], powers_db=[0, -3])
    noise = AWGNChannel()
    estimator = LSEstimator(pilots.pilot_indices, n_subcarriers)
    equalizer = ZeroForcingEqualizer()
    
    # QPSK
    mod_qpsk = QAMModulator(M=4)
    n_bits_qpsk = n_ofdm_symbols * len(pilots.data_indices) * mod_qpsk.k
    tx_bits_qpsk = np.random.randint(0, 2, n_bits_qpsk)
    tx_serial_qpsk = ofdm.modulate(pilots.insert(mod_qpsk.modulate(tx_bits_qpsk).reshape(n_ofdm_symbols, -1))).flatten()
    faded_qpsk, _ = channel.apply(tx_serial_qpsk)
    
    # 16QAM
    mod_16qam = QAMModulator(M=16)
    n_bits_16qam = n_ofdm_symbols * len(pilots.data_indices) * mod_16qam.k
    tx_bits_16qam = np.random.randint(0, 2, n_bits_16qam)
    tx_serial_16qam = ofdm.modulate(pilots.insert(mod_16qam.modulate(tx_bits_16qam).reshape(n_ofdm_symbols, -1))).flatten()
    faded_16qam, _ = channel.apply(tx_serial_16qam)
    
    for snr in snr_range:
        # QPSK Rx
        rx_grid_qpsk = ofdm.demodulate(noise.apply(faded_qpsk, snr).reshape(n_ofdm_symbols, -1))
        p_qpsk, d_qpsk = pilots.extract(rx_grid_qpsk)
        rx_bits_qpsk = mod_qpsk.demodulate(equalizer.equalize(d_qpsk, estimator.estimate(p_qpsk), pilots.data_indices).flatten())
        ber_qpsk.append(calculate_ber(tx_bits_qpsk, rx_bits_qpsk))

        # 16QAM Rx
        rx_grid_16qam = ofdm.demodulate(noise.apply(faded_16qam, snr).reshape(n_ofdm_symbols, -1))
        p_16qam, d_16qam = pilots.extract(rx_grid_16qam)
        rx_bits_16qam = mod_16qam.demodulate(equalizer.equalize(d_16qam, estimator.estimate(p_16qam), pilots.data_indices).flatten())
        ber_16qam.append(calculate_ber(tx_bits_16qam, rx_bits_16qam))
        
    # Apply BER floor so zero-error SNR points stay visible on log axis
    n_bits_qpsk = n_ofdm_symbols * len(pilots.data_indices) * mod_qpsk.k
    n_bits_16qam = n_ofdm_symbols * len(pilots.data_indices) * mod_16qam.k
    ber_qpsk  = [max(b, 0.5 / n_bits_qpsk)  for b in ber_qpsk]
    ber_16qam = [max(b, 0.5 / n_bits_16qam) for b in ber_16qam]

    plt.figure()
    plt.semilogy(snr_range, ber_qpsk, 'o-', label='QPSK')
    plt.semilogy(snr_range, ber_16qam, 's--', label='16-QAM')
    plt.title("BER vs SNR: Modulation Schemes\n(TDL Rayleigh fading, LS estimation, ZF equaliser)")
    # The AWGNChannel uses Es/N0 (energy-per-symbol / noise).  To convert to
    # Eb/N0 for textbook comparison, subtract 10*log10(k) where k=log2(M):
    #   QPSK:   Es/N0 = Eb/N0 + 3 dB
    #   16-QAM: Es/N0 = Eb/N0 + 6 dB
    plt.xlabel("SNR — Es/N0 (dB)")
    plt.ylabel("Bit Error Rate (BER)")
    plt.grid(True, which="both", ls="-")
    plt.legend()
    plt.tight_layout()
    plt.savefig("compare_modulation.png", dpi=150)
    print("Saved plot to compare_modulation.png")

if __name__ == "__main__":
    run_experiment()
