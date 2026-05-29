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
    """
    Sweeps SNR from 0–30 dB and plots BER for LS vs MMSE channel estimation.

    Monte Carlo averaging over N_TRIALS independent noise realisations is used
    so that BER at high SNR is estimated from enough error events to produce a
    statistically reliable curve rather than a single lucky zero-error trial
    that gets silently dropped by matplotlib's semilogy.

    A BER floor of 0.5 / total_bits is applied so that zero-error trials appear
    at the bottom of the log-scale axis rather than disappearing from the plot.
    """
    snr_range = np.arange(0, 31, 5)

    # Increase symbol count and add Monte Carlo trials to collect enough bit
    # errors at all SNR points, especially above 15 dB.
    N_TRIALS = 5           # independent noise realisations per SNR point
    n_subcarriers = 64
    n_ofdm_symbols = 200   # was 50 — raised to lower variance at high SNR

    modulator = QAMModulator(M=4)  # QPSK
    ofdm = OFDM(n_subcarriers=n_subcarriers)
    pilots = PilotInserter(n_subcarriers=n_subcarriers)
    channel = TDLChannel(sample_rate=1e6, delays=[0, 1e-6], powers_db=[0, -3])
    noise = AWGNChannel()

    estimator_ls = LSEstimator(pilots.pilot_indices, n_subcarriers)
    equalizer = ZeroForcingEqualizer()

    n_bits = n_ofdm_symbols * len(pilots.data_indices) * modulator.k

    # Pre-generate and fade the signal once; noise is varied per SNR trial
    tx_bits = np.random.randint(0, 2, n_bits)
    tx_symbols = modulator.modulate(tx_bits).reshape(n_ofdm_symbols, -1)
    tx_grid = pilots.insert(tx_symbols)
    tx_serial = ofdm.modulate(tx_grid).flatten()
    faded_signal, _ = channel.apply(tx_serial)

    ber_ls_list = []
    ber_mmse_list = []

    for snr in snr_range:
        # Accumulate errors over N_TRIALS to get a stable BER estimate
        err_ls = err_mmse = n_bits_total = 0

        for _ in range(N_TRIALS):
            # Re-generate tx bits and signal for each trial (independent fades
            # would be more rigorous, but varying noise at fixed fading is the
            # standard single-channel-realisation approach used here)
            rx_serial = noise.apply(faded_signal, snr)
            rx_grid = ofdm.demodulate(rx_serial.reshape(n_ofdm_symbols, -1))
            rx_pilots, rx_data = pilots.extract(rx_grid)

            # LS estimation
            H_est_ls = estimator_ls.estimate(rx_pilots)
            eq_data_ls = equalizer.equalize(rx_data, H_est_ls, pilots.data_indices)
            rx_bits_ls = modulator.demodulate(eq_data_ls.flatten())
            err_ls += int(np.sum(tx_bits != rx_bits_ls[:len(tx_bits)]))

            # MMSE (diagonal LMMSE) estimation
            estimator_mmse = MMSEEstimator(pilots.pilot_indices, n_subcarriers, snr_db=snr)
            H_est_mmse = estimator_mmse.estimate(rx_pilots)
            eq_data_mmse = equalizer.equalize(rx_data, H_est_mmse, pilots.data_indices)
            rx_bits_mmse = modulator.demodulate(eq_data_mmse.flatten())
            err_mmse += int(np.sum(tx_bits != rx_bits_mmse[:len(tx_bits)]))

            n_bits_total += n_bits

        # Apply a BER floor so zero-error SNR points remain visible on a log axis
        # instead of being silently dropped by semilogy.
        ber_floor = 0.5 / n_bits_total
        ber_ls_list.append(max(err_ls / n_bits_total, ber_floor))
        ber_mmse_list.append(max(err_mmse / n_bits_total, ber_floor))

    plt.figure()
    plt.semilogy(snr_range, ber_ls_list, "o-", label="LS Estimation")
    plt.semilogy(snr_range, ber_mmse_list, "s--", label="MMSE Estimation (LMMSE)")
    plt.title("BER vs SNR: LS vs MMSE Channel Estimation\n(QPSK, TDL Rayleigh fading, ZF equaliser)")
    plt.xlabel("SNR — Es/N0 (dB)")
    plt.ylabel("Bit Error Rate (BER)")
    plt.grid(True, which="both", ls="-")
    plt.legend()
    plt.tight_layout()
    plt.savefig("compare_estimators.png", dpi=150)
    print("Saved plot to compare_estimators.png")


if __name__ == "__main__":
    run_experiment()
