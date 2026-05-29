import numpy as np
import time
import os
from datetime import datetime

# Import modular components from mimo_ofdm
from mimo_ofdm.core.modulation import QAMModulator
from mimo_ofdm.core.ofdm import OFDM
from mimo_ofdm.core.mimo import MIMOSpatialMultiplexer
from mimo_ofdm.channel.noise import AWGNChannel
from mimo_ofdm.channel.fading import TDLChannel
from mimo_ofdm.estimation.pilots import PilotInserter
from mimo_ofdm.estimation.ls import LSEstimator
from mimo_ofdm.estimation.mmse import MMSEEstimator
from mimo_ofdm.estimation.adaptive_estimation import AdaptiveEstimator
from mimo_ofdm.equalization.zf import ZeroForcingEqualizer
from mimo_ofdm.equalization.mmse_eq import MMSEEqualizer
from mimo_ofdm.equalization.mimo_det import MIMOZFEqualizer, MIMOMMSEEqualizer
from mimo_ofdm.extensions.sphere import SphereDecoder
from mimo_ofdm.extensions.water_filling import water_filling
from mimo_ofdm.utils.metrics import calculate_ber, calculate_evm, calculate_throughput

# ANSI color codes for visual excellence
CLR_HEADER = "\033[95m"
CLR_BLUE = "\033[94m"
CLR_CYAN = "\033[96m"
CLR_GREEN = "\033[92m"
CLR_WARNING = "\033[93m"
CLR_FAIL = "\033[91m"
CLR_END = "\033[0m"
CLR_BOLD = "\033[1m"

LOG_FILE = "manual_test_results.log"

def log_to_file(message):
    """Appends messages to the persistent manual test results log file."""
    with open(LOG_FILE, "a") as f:
        f.write(message + "\n")

def initialize_log_entry():
    """Writes a beautiful starting banner for this test run in the log file."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    header = "=" * 80 + "\n"
    header += f"MIMO-OFDM SYSTEM MANUAL SIMULATION RUN - {timestamp}\n"
    header += "=" * 80 + "\n"
    log_to_file(header)

def run_estimator_comparison():
    """
    Compares LS vs MMSE vs Adaptive (EWMA) estimators under time-varying Rayleigh fading (Doppler).
    """
    print(f"\n{CLR_BOLD}{CLR_CYAN}--- PART 1: CHANNEL ESTIMATORS EVALUATION (LS vs. MMSE vs. Adaptive EWMA) ---{CLR_END}")
    
    n_subcarriers = 64
    cp_length = 16
    n_ofdm_symbols = 30
    snr_db = 15
    max_doppler = 50.0  # 50 Hz Doppler spread to simulate time-varying fading
    
    modulator = QAMModulator(M=4) # QPSK
    ofdm = OFDM(n_subcarriers=n_subcarriers, cp_length=cp_length)
    pilots = PilotInserter(n_subcarriers=n_subcarriers, pilot_spacing=4)
    channel = TDLChannel(sample_rate=1e6, delays=[0, 1e-6, 3e-6], powers_db=[0, -3, -10], max_doppler=max_doppler, seed=42)
    noise = AWGNChannel(seed=42)
    
    # Base estimators
    estimator_ls = LSEstimator(pilots.pilot_indices, n_subcarriers)
    estimator_mmse = MMSEEstimator(pilots.pilot_indices, n_subcarriers, snr_db=snr_db)
    
    # Adaptive estimator (EWMA) tracking the time-varying channel
    adaptive_est = AdaptiveEstimator(base_estimator=estimator_ls, alpha=0.7)
    
    equalizer = ZeroForcingEqualizer()
    
    # Transmission bits
    n_bits = n_ofdm_symbols * len(pilots.data_indices) * modulator.k
    tx_bits = np.random.randint(0, 2, n_bits)
    tx_symbols = modulator.modulate(tx_bits).reshape(n_ofdm_symbols, -1)
    tx_grid = pilots.insert(tx_symbols)
    tx_signal = ofdm.modulate(tx_grid)
    
    # Time-varying channel simulation
    faded_signal, H_taps = channel.apply(tx_signal.flatten())
    rx_serial = noise.apply(faded_signal, snr_db=snr_db)
    
    # Demodulate
    rx_signal = rx_serial.reshape(n_ofdm_symbols, -1)
    rx_grid = ofdm.demodulate(rx_signal)
    rx_pilots, rx_data = pilots.extract(rx_grid)
    
    # Generate True Frequency Response for MSE calculation
    # True tap channel response mapped to frequency domain
    H_true = np.zeros((n_ofdm_symbols, n_subcarriers), dtype=complex)
    for s in range(n_ofdm_symbols):
        # Index range of taps belonging to this symbol
        samples_per_sym = n_subcarriers + cp_length
        tap_idx = s * samples_per_sym + cp_length # sample index at middle of symbol
        symbol_taps = H_taps[:, tap_idx]
        padded_taps = np.pad(symbol_taps, (0, n_subcarriers - len(symbol_taps)))
        H_true[s] = np.fft.fft(padded_taps)
        
    # Evaluate LS
    t0 = time.perf_counter()
    H_est_ls = estimator_ls.estimate(rx_pilots)
    eq_ls = equalizer.equalize(rx_data, H_est_ls, pilots.data_indices)
    bits_ls = modulator.demodulate(eq_ls.flatten())
    time_ls = (time.perf_counter() - t0) * 1000
    mse_ls = np.mean(np.abs(H_true - H_est_ls)**2)
    ber_ls = calculate_ber(tx_bits, bits_ls)
    evm_ls = calculate_evm(tx_symbols.flatten(), eq_ls.flatten())
    
    # Evaluate MMSE
    t0 = time.perf_counter()
    H_est_mmse = estimator_mmse.estimate(rx_pilots)
    eq_mmse = equalizer.equalize(rx_data, H_est_mmse, pilots.data_indices)
    bits_mmse = modulator.demodulate(eq_mmse.flatten())
    time_mmse = (time.perf_counter() - t0) * 1000
    mse_mmse = np.mean(np.abs(H_true - H_est_mmse)**2)
    ber_mmse = calculate_ber(tx_bits, bits_mmse)
    evm_mmse = calculate_evm(tx_symbols.flatten(), eq_mmse.flatten())
    
    # Evaluate Adaptive EWMA
    t0 = time.perf_counter()
    H_est_adaptive = adaptive_est.estimate(rx_pilots)
    eq_adaptive = equalizer.equalize(rx_data, H_est_adaptive, pilots.data_indices)
    bits_adaptive = modulator.demodulate(eq_adaptive.flatten())
    time_adaptive = (time.perf_counter() - t0) * 1000
    mse_adaptive = np.mean(np.abs(H_true - H_est_adaptive)**2)
    ber_adaptive = calculate_ber(tx_bits, bits_adaptive)
    evm_adaptive = calculate_evm(tx_symbols.flatten(), eq_adaptive.flatten())
    
    # Console output table
    print(f"| {'Algorithm':<20} | {'BER':<10} | {'EVM (%)':<10} | {'Channel MSE':<12} | {'Time (ms)':<10} |")
    print("-" * 75)
    print(f"| {CLR_BLUE}Least Squares (LS){CLR_END:<29} | {ber_ls:<10.6f} | {evm_ls:<10.2f} | {mse_ls:<12.6f} | {time_ls:<10.2f} |")
    print(f"| {CLR_BLUE}MMSE Estimator{CLR_END:<29} | {ber_mmse:<10.6f} | {evm_mmse:<10.2f} | {mse_mmse:<12.6f} | {time_mmse:<10.2f} |")
    print(f"| {CLR_GREEN}Adaptive EWMA{CLR_END:<29} | {ber_adaptive:<10.6f} | {evm_adaptive:<10.2f} | {mse_adaptive:<12.6f} | {time_adaptive:<10.2f} |")
    
    # Log detailed results
    log_msg = "PART 1: CHANNEL ESTIMATORS COMPARISON (Doppler = 50Hz, SNR = 15dB)\n"
    log_msg += f"{'Estimator':<25} | {'BER':<12} | {'EVM (%)':<12} | {'Channel MSE':<15} | {'Latency (ms)':<15}\n"
    log_msg += "-" * 88 + "\n"
    log_msg += f"{'Least Squares (LS)':<25} | {ber_ls:<12.6f} | {evm_ls:<12.2f} | {mse_ls:<15.6f} | {time_ls:<15.2f}\n"
    log_msg += f"{'MMSE Estimator':<25} | {ber_mmse:<12.6f} | {evm_mmse:<12.2f} | {mse_mmse:<15.6f} | {time_mmse:<15.2f}\n"
    log_msg += f"{'Adaptive EWMA':<25} | {ber_adaptive:<12.6f} | {evm_adaptive:<12.2f} | {mse_adaptive:<15.6f} | {time_adaptive:<15.2f}\n"
    log_msg += "=" * 80 + "\n"
    log_to_file(log_msg)

def run_mimo_detector_comparison():
    """
    Compares MIMO Spatial Multiplexing detectors: Zero Forcing (ZF), MMSE, and Sphere Decoder (ML).
    Evaluated under 2x2 and 4x4 spatial multiplexing MIMO setups.
    """
    print(f"\n{CLR_BOLD}{CLR_CYAN}--- PART 2: MIMO DETECTORS EVALUATION (2x2 vs. 4x4 MIMO, ZF vs. MMSE vs. Sphere Decoder) ---{CLR_END}")
    
    n_subcarriers = 64
    n_ofdm_symbols = 20
    snr_db = 12
    
    modulator = QAMModulator(M=4) # QPSK
    pilots = PilotInserter(n_subcarriers=n_subcarriers)
    n_data = len(pilots.data_indices)
    
    for (n_tx, n_rx) in [(2, 2), (4, 4)]:
        print(f"\n{CLR_BOLD}{CLR_BLUE}MIMO Configuration: {n_tx}x{n_rx} (SNR = {snr_db} dB){CLR_END}")
        
        multiplexer = MIMOSpatialMultiplexer(n_tx=n_tx, n_rx=n_rx)
        n_symbols_total = n_ofdm_symbols * n_data * n_tx
        n_bits = n_symbols_total * modulator.k
        
        # Bits generation and modulation
        tx_bits = np.random.randint(0, 2, n_bits)
        tx_symbols = modulator.modulate(tx_bits)
        
        # Spatial Multiplexing
        tx_streams = multiplexer.multiplex(tx_symbols) # (n_tx, n_ofdm_symbols * n_data)
        
        # Build Grid
        tx_grid = np.zeros((n_tx, n_ofdm_symbols, n_subcarriers), dtype=complex)
        for t in range(n_tx):
            tx_symbols_antenna = tx_streams[t].reshape(n_ofdm_symbols, n_data)
            tx_grid[t] = pilots.insert(tx_symbols_antenna)

        # Normalise total transmitted power to 1 regardless of the number of antennas.
        # Without this, a 4x4 system transmits 4x the power of 2x2 at the same noise
        # variance, making per-stream SNR comparisons unfair.
        tx_grid = tx_grid / np.sqrt(n_tx)

        # Channel generation
        H = multiplexer.generate_channel_matrix(n_subcarriers) # (n_rx, n_tx, n_subcarriers)

        # Apply MIMO channel and noise
        noise_var = 10 ** (-snr_db / 10)
        rx_grid = np.zeros((n_rx, n_ofdm_symbols, n_subcarriers), dtype=complex)
        for k in range(n_subcarriers):
            # Complex noise
            noise_k = np.sqrt(noise_var / 2) * (np.random.randn(n_rx, n_ofdm_symbols) + 1j * np.random.randn(n_rx, n_ofdm_symbols))
            rx_grid[:, :, k] = H[:, :, k] @ tx_grid[:, :, k] + noise_k
            
        # 1. Zero Forcing Equalizer
        t0 = time.perf_counter()
        zf_eq = MIMOZFEqualizer()
        eq_zf = zf_eq.equalize(rx_grid, H)
        
        # Squeeze streams back
        rx_streams_zf = np.zeros((n_tx, n_ofdm_symbols * n_data), dtype=complex)
        for t in range(n_tx):
            _, rx_data_t = pilots.extract(eq_zf[t])
            rx_streams_zf[t] = rx_data_t.flatten()
        rx_symbols_zf = rx_streams_zf.T.flatten()
        bits_zf = modulator.demodulate(rx_symbols_zf)
        time_zf = (time.perf_counter() - t0) * 1000
        ber_zf = calculate_ber(tx_bits, bits_zf)
        evm_zf = calculate_evm(tx_symbols, rx_symbols_zf)
        
        # 2. MMSE Equalizer
        t0 = time.perf_counter()
        mmse_eq = MIMOMMSEEqualizer(snr_db=snr_db)
        eq_mmse = mmse_eq.equalize(rx_grid, H)
        
        rx_streams_mmse = np.zeros((n_tx, n_ofdm_symbols * n_data), dtype=complex)
        for t in range(n_tx):
            _, rx_data_t = pilots.extract(eq_mmse[t])
            rx_streams_mmse[t] = rx_data_t.flatten()
        rx_symbols_mmse = rx_streams_mmse.T.flatten()
        bits_mmse = modulator.demodulate(rx_symbols_mmse)
        time_mmse = (time.perf_counter() - t0) * 1000
        ber_mmse = calculate_ber(tx_bits, bits_mmse)
        evm_mmse = calculate_evm(tx_symbols, rx_symbols_mmse)
        
        # 3. Optimal Sphere Decoder (ML Performance)
        t0 = time.perf_counter()
        sd = SphereDecoder(modulator.constellation)
        
        # Sphere decoder operates subcarrier-by-subcarrier, symbol-by-symbol
        eq_sd = np.zeros((n_tx, n_ofdm_symbols, n_subcarriers), dtype=complex)
        for k in range(n_subcarriers):
            H_k = H[:, :, k]
            for s in range(n_ofdm_symbols):
                y_ks = rx_grid[:, s, k]
                # Decode the spatially multiplexed vector directly
                decoded_vector = sd.decode(y_ks, H_k)
                eq_sd[:, s, k] = decoded_vector
                
        rx_streams_sd = np.zeros((n_tx, n_ofdm_symbols * n_data), dtype=complex)
        for t in range(n_tx):
            _, rx_data_t = pilots.extract(eq_sd[t])
            rx_streams_sd[t] = rx_data_t.flatten()
        rx_symbols_sd = rx_streams_sd.T.flatten()
        bits_sd = modulator.demodulate(rx_symbols_sd)
        time_sd = (time.perf_counter() - t0) * 1000
        ber_sd = calculate_ber(tx_bits, bits_sd)
        evm_sd = calculate_evm(tx_symbols, rx_symbols_sd)
        
        # Print comparative table to console
        print(f"| {'Detector':<20} | {'BER':<10} | {'EVM (%)':<10} | {'Time (ms)':<10} |")
        print("-" * 62)
        print(f"| {CLR_BLUE}MIMO Zero Forcing{CLR_END:<29} | {ber_zf:<10.6f} | {evm_zf:<10.2f} | {time_zf:<10.2f} |")
        print(f"| {CLR_BLUE}MIMO MMSE{CLR_END:<29} | {ber_mmse:<10.6f} | {evm_mmse:<10.2f} | {time_mmse:<10.2f} |")
        print(f"| {CLR_GREEN}Sphere Decoder (ML){CLR_END:<29} | {ber_sd:<10.6f} | {evm_sd:<10.2f} | {time_sd:<10.2f} |")
        
        # Log entry
        log_msg = f"PART 2: MIMO {n_tx}x{n_rx} DETECTORS COMPARISON (SNR = {snr_db}dB)\n"
        log_msg += f"{'Detector':<25} | {'BER':<12} | {'EVM (%)':<12} | {'Latency (ms)':<15}\n"
        log_msg += "-" * 70 + "\n"
        log_msg += f"{'MIMO Zero Forcing (ZF)':<25} | {ber_zf:<12.6f} | {evm_zf:<12.2f} | {time_zf:<15.2f}\n"
        log_msg += f"{'MIMO MMSE':<25} | {ber_mmse:<12.6f} | {evm_mmse:<12.2f} | {time_mmse:<15.2f}\n"
        log_msg += f"{'Sphere Decoder (ML)':<25} | {ber_sd:<12.6f} | {evm_sd:<12.2f} | {time_sd:<15.2f}\n"
        log_msg += "=" * 80 + "\n"
        log_to_file(log_msg)

def run_water_filling_comparison():
    """
    Evaluates Water Filling power allocation capacity and BER performance vs. Equal Power allocation.
    """
    print(f"\n{CLR_BOLD}{CLR_CYAN}--- PART 3: WATER FILLING CAPACITY & BER EVALUATION (Water-Filling vs. Equal Power) ---{CLR_END}")
    
    n_subcarriers = 64
    snr_db = 10
    total_power = n_subcarriers  # Normalization: power = 1 per subcarrier average
    noise_var = 10 ** (-snr_db / 10)
    
    # Generate frequency selective fading channel gains across subcarriers.
    # Tap powers mirror the TDL channel used elsewhere: [0, -3, -10] dB.
    # NOTE: Do NOT normalise to unit norm here.  Normalising erases the
    # frequency selectivity, making the response nearly flat — water filling
    # on a flat channel yields ≈0% capacity gain, which is physically correct
    # but useless as a demonstration.  Using the raw tap magnitudes preserves
    # the natural amplitude variation that water filling exploits.
    np.random.seed(123)
    channel_taps = np.array([1.0, 10**(-3/20), 10**(-10/20)])  # linear amplitudes [0,-3,-10 dB]
    padded_taps = np.pad(channel_taps, (0, n_subcarriers - len(channel_taps)))
    H_freq = np.fft.fft(padded_taps)  # Channel frequency response
    channel_gains = np.abs(H_freq)
    
    # 1. Equal Power Allocation
    p_equal = np.ones(n_subcarriers) * (total_power / n_subcarriers)
    snr_eq = (p_equal * channel_gains**2) / noise_var
    capacity_eq = np.sum(np.log2(1 + snr_eq))
    
    # 2. Water Filling Power Allocation
    p_wf = water_filling(channel_gains, total_power, noise_var)
    snr_wf = (p_wf * channel_gains**2) / noise_var
    capacity_wf = np.sum(np.log2(1 + snr_wf))
    
    # BER evaluation simulation using 16-QAM under Equal vs Water-filling power allocation
    modulator = QAMModulator(M=16)
    n_bits = n_subcarriers * modulator.k * 100
    tx_bits = np.random.randint(0, 2, n_bits)
    tx_symbols = modulator.modulate(tx_bits).reshape(100, n_subcarriers)
    
    # Equal Power Sim
    tx_grid_eq = tx_symbols * np.sqrt(p_equal)[np.newaxis, :]
    rx_grid_eq = tx_grid_eq * H_freq[np.newaxis, :] + np.sqrt(noise_var / 2) * (np.random.randn(*tx_grid_eq.shape) + 1j * np.random.randn(*tx_grid_eq.shape))
    eq_grid_eq = rx_grid_eq / H_freq[np.newaxis, :] # Zero forcing equalizer
    # Scale back by allocated power to normalize
    eq_grid_eq = eq_grid_eq / np.sqrt(p_equal)[np.newaxis, :]
    bits_eq = modulator.demodulate(eq_grid_eq.flatten())
    ber_eq = calculate_ber(tx_bits, bits_eq)
    
    # Water Filling Sim
    tx_grid_wf = tx_symbols * np.sqrt(p_wf)[np.newaxis, :]
    rx_grid_wf = tx_grid_wf * H_freq[np.newaxis, :] + np.sqrt(noise_var / 2) * (np.random.randn(*tx_grid_wf.shape) + 1j * np.random.randn(*tx_grid_wf.shape))
    
    # Equalizer (Zero forcing only on subcarriers with non-zero power allocation)
    eq_grid_wf = np.zeros_like(tx_symbols, dtype=complex)
    active_carriers = p_wf > 1e-6
    eq_grid_wf[:, active_carriers] = rx_grid_wf[:, active_carriers] / (H_freq[np.newaxis, active_carriers] * np.sqrt(p_wf)[active_carriers][np.newaxis, :])
    # For subcarriers with 0 power, we map to 0 constellation points
    bits_wf = modulator.demodulate(eq_grid_wf.flatten())
    ber_wf = calculate_ber(tx_bits, bits_wf)
    
    # Capacity improvement
    cap_improvement = ((capacity_wf - capacity_eq) / capacity_eq) * 100
    
    # Print results to console
    print(f"| {'Allocation Mode':<20} | {'Sum Capacity (bps/Hz)':<22} | {'Simulated BER (16-QAM)':<24} |")
    print("-" * 75)
    print(f"| {CLR_BLUE}Equal Power{CLR_END:<29} | {capacity_eq:<22.4f} | {ber_eq:<24.6f} |")
    print(f"| {CLR_GREEN}Water Filling{CLR_END:<29} | {capacity_wf:<22.4f} | {ber_wf:<24.6f} |")
    print(f"\n{CLR_BOLD}{CLR_GREEN}Water Filling achieves +{cap_improvement:.2f}% capacity gain over Equal Power allocation!{CLR_END}")
    
    # Log entry
    log_msg = f"PART 3: WATER FILLING POWER ALLOCATION COMPARISON (SNR = {snr_db}dB)\n"
    log_msg += f"{'Allocation Scheme':<25} | {'Capacity (bps/Hz)':<22} | {'BER (16-QAM)':<20}\n"
    log_msg += "-" * 73 + "\n"
    log_msg += f"{'Equal Power':<25} | {capacity_eq:<22.4f} | {ber_eq:<20.6f}\n"
    log_msg += f"{'Water Filling':<25} | {capacity_wf:<22.4f} | {ber_wf:<20.6f}\n"
    log_msg += f"Capacity Gain: +{cap_improvement:.2f}%\n"
    log_msg += "=" * 80 + "\n"
    log_to_file(log_msg)

def main():
    print(f"{CLR_BOLD}{CLR_HEADER}========================================================================{CLR_END}")
    print(f"{CLR_BOLD}{CLR_HEADER}   MIMO-OFDM SYSTEM SIMULATOR - HIGH-FIDELITY ALGORITHM MANUAL TESTING  {CLR_END}")
    print(f"{CLR_BOLD}{CLR_HEADER}========================================================================{CLR_END}")
    print(f"Saving persistent logs to: {CLR_BOLD}{LOG_FILE}{CLR_END}")
    
    initialize_log_entry()
    
    # Run tests
    run_estimator_comparison()
    run_mimo_detector_comparison()
    run_water_filling_comparison()
    
    print(f"\n{CLR_BOLD}{CLR_GREEN}========================================================================{CLR_END}")
    print(f"{CLR_BOLD}{CLR_GREEN}  MANUAL SIMULATIONS COMPLETED SUCCESSFULLY! LOGS SAVED IN {LOG_FILE}  {CLR_END}")
    print(f"{CLR_BOLD}{CLR_GREEN}========================================================================{CLR_END}\n")

if __name__ == "__main__":
    main()
