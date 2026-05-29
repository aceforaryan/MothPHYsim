import numpy as np
import matplotlib.pyplot as plt
from mimo_ofdm.core.modulation import QAMModulator
from mimo_ofdm.core.ofdm import OFDM
from mimo_ofdm.core.mimo import MIMOSpatialMultiplexer
from mimo_ofdm.estimation.pilots import PilotInserter
from mimo_ofdm.equalization.mimo_det import MIMOZFEqualizer, MIMOMMSEEqualizer
from mimo_ofdm.utils.metrics import calculate_ber

def simulate_mimo(n_tx, n_rx, snr_range, demod_type='ZF'):
    """
    Simulates a MIMO-OFDM system over a range of SNR values.
    Returns a list of BERs for each SNR.
    """
    n_subcarriers = 64
    n_ofdm_symbols = 50
    
    modulator = QAMModulator(M=4) # QPSK
    ofdm = OFDM(n_subcarriers=n_subcarriers)
    pilots = PilotInserter(n_subcarriers=n_subcarriers)
    multiplexer = MIMOSpatialMultiplexer(n_tx=n_tx, n_rx=n_rx)
    
    n_data_subcarriers = len(pilots.data_indices)
    
    # Calculate total number of data symbols needed
    n_symbols_total = n_ofdm_symbols * n_data_subcarriers * n_tx
    n_bits = n_symbols_total * modulator.k
    
    # Generate random transmit bits
    tx_bits = np.random.randint(0, 2, n_bits)
    tx_symbols = modulator.modulate(tx_bits)
    
    # Multiplex symbols across transmit antennas
    tx_streams = multiplexer.multiplex(tx_symbols) # (n_tx, n_ofdm_symbols * n_data_subcarriers)
    
    # Build the transmitted grid for each antenna
    tx_grid = np.zeros((n_tx, n_ofdm_symbols, n_subcarriers), dtype=complex)
    for t in range(n_tx):
        tx_symbols_antenna = tx_streams[t].reshape(n_ofdm_symbols, n_data_subcarriers)
        tx_grid[t] = pilots.insert(tx_symbols_antenna)

    # Normalise total transmitted power to 1 regardless of the number of antennas.
    # Without this, a 4x4 system transmits 4x the power of a 2x2 system at the same
    # noise_var, making per-stream SNR comparisons unfair and causing ZF noise
    # amplification to dominate for larger arrays.
    tx_grid = tx_grid / np.sqrt(n_tx)
    
    # Generate MIMO channel matrix H (n_rx, n_tx, n_subcarriers)
    H = multiplexer.generate_channel_matrix(n_subcarriers)
    
    bers = []
    for snr in snr_range:
        noise_var = 10 ** (-snr / 10)
        
        # Apply channel and noise subcarrier by subcarrier
        rx_grid = np.zeros((n_rx, n_ofdm_symbols, n_subcarriers), dtype=complex)
        for k in range(n_subcarriers):
            H_k = H[:, :, k] # (n_rx, n_tx)
            # Signal contribution
            rx_signal_k = H_k @ tx_grid[:, :, k] # (n_rx, n_ofdm_symbols)
            # Complex white Gaussian noise contribution
            noise_k = np.sqrt(noise_var / 2) * (np.random.randn(n_rx, n_ofdm_symbols) + 
                                               1j * np.random.randn(n_rx, n_ofdm_symbols))
            rx_grid[:, :, k] = rx_signal_k + noise_k
            
        # Perform equalization/detection
        if demod_type == 'ZF':
            equalizer = MIMOZFEqualizer()
            eq_grid = equalizer.equalize(rx_grid, H)
        elif demod_type == 'MMSE':
            equalizer = MIMOMMSEEqualizer(snr_db=snr)
            eq_grid = equalizer.equalize(rx_grid, H)
        else:
            raise ValueError(f"Unknown demodulation type: {demod_type}")
            
        # Reconstruct streams and recover symbols
        rx_streams = np.zeros((n_tx, n_ofdm_symbols * n_data_subcarriers), dtype=complex)
        for t in range(n_tx):
            _, rx_data_t = pilots.extract(eq_grid[t])
            rx_streams[t] = rx_data_t.flatten()
            
        # Reconstruct flat complex symbols array
        rx_symbols = rx_streams.T.flatten()
        rx_bits = modulator.demodulate(rx_symbols)
        
        # Calculate Bit Error Rate
        ber = calculate_ber(tx_bits, rx_bits)
        bers.append(ber)
        
    return bers

def run_experiment():
    print("Starting MIMO Spatial Multiplexing Experiment...")
    snr_range = np.arange(0, 31, 5)
    
    print("Simulating 2x2 MIMO with ZF detector...")
    ber_2x2_zf = simulate_mimo(n_tx=2, n_rx=2, snr_range=snr_range, demod_type='ZF')
    
    print("Simulating 2x2 MIMO with MMSE detector...")
    ber_2x2_mmse = simulate_mimo(n_tx=2, n_rx=2, snr_range=snr_range, demod_type='MMSE')
    
    print("Simulating 4x4 MIMO with ZF detector...")
    ber_4x4_zf = simulate_mimo(n_tx=4, n_rx=4, snr_range=snr_range, demod_type='ZF')
    
    print("Simulating 4x4 MIMO with MMSE detector...")
    ber_4x4_mmse = simulate_mimo(n_tx=4, n_rx=4, snr_range=snr_range, demod_type='MMSE')
    
    # Plot results
    plt.figure(figsize=(10, 6))
    plt.semilogy(snr_range, ber_2x2_zf, 'o-', color='#e056fd', label='2x2 MIMO (ZF)')
    plt.semilogy(snr_range, ber_2x2_mmse, 's--', color='#686de0', label='2x2 MIMO (MMSE)')
    plt.semilogy(snr_range, ber_4x4_zf, 'd-', color='#ff7979', label='4x4 MIMO (ZF)')
    plt.semilogy(snr_range, ber_4x4_mmse, 'x--', color='#30336b', label='4x4 MIMO (MMSE)')
    
    plt.title("BER vs SNR: MIMO Spatial Multiplexing (2x2 vs 4x4)", fontsize=14, fontweight='bold')
    plt.xlabel("SNR — Es/N0 (dB)", fontsize=12)
    plt.ylabel("Bit Error Rate (BER)", fontsize=12)
    plt.grid(True, which="both", ls="-", alpha=0.5)
    plt.legend(fontsize=11)
    
    # Modern style adjustments
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    
    plt.savefig("compare_mimo.png", dpi=300, bbox_inches='tight')
    print("Saved plot to compare_mimo.png")

if __name__ == "__main__":
    run_experiment()
