import numpy as np

from mimo_ofdm.core.modulation import QAMModulator
from mimo_ofdm.core.ofdm import OFDM
from mimo_ofdm.channel.noise import AWGNChannel
from mimo_ofdm.channel.fading import TDLChannel
from mimo_ofdm.estimation.pilots import PilotInserter
from mimo_ofdm.estimation.ls import LSEstimator
from mimo_ofdm.equalization.zf import ZeroForcingEqualizer
from mimo_ofdm.utils.metrics import calculate_ber
from mimo_ofdm.utils.visualize import explainability_pipeline

def main():
    print("MIMO-OFDM Simulation Started...")
    
    # 1. System Parameters
    n_subcarriers = 64
    cp_length = 16
    snr_db = 20
    n_ofdm_symbols = 10
    
    # 2. Components
    modulator = QAMModulator(M=16) # 16-QAM
    ofdm = OFDM(n_subcarriers=n_subcarriers, cp_length=cp_length)
    pilots = PilotInserter(n_subcarriers=n_subcarriers, pilot_spacing=4)
    channel = TDLChannel(sample_rate=1e6, delays=[0, 1e-6, 3e-6], powers_db=[0, -3, -10])
    noise = AWGNChannel()
    estimator = LSEstimator(pilot_indices=pilots.pilot_indices, n_subcarriers=n_subcarriers)
    equalizer = ZeroForcingEqualizer()

    # 3. Transmission Pipeline
    n_bits = n_ofdm_symbols * len(pilots.data_indices) * modulator.k
    tx_bits = np.random.randint(0, 2, n_bits)
    
    tx_symbols = modulator.modulate(tx_bits)
    tx_symbols_reshaped = tx_symbols.reshape(n_ofdm_symbols, -1)
    
    tx_grid = pilots.insert(tx_symbols_reshaped)
    tx_signal = ofdm.modulate(tx_grid)
    tx_serial = tx_signal.flatten()
    
    # 4. Channel
    faded_signal, H_taps = channel.apply(tx_serial)
    rx_serial = noise.apply(faded_signal, snr_db)
    
    # 5. Reception Pipeline
    rx_signal = rx_serial.reshape(n_ofdm_symbols, -1)
    rx_grid = ofdm.demodulate(rx_signal)
    
    rx_pilots, rx_data = pilots.extract(rx_grid)
    
    H_est = estimator.estimate(rx_pilots)
    eq_data = equalizer.equalize(rx_data, H_est, pilots.data_indices)
    
    rx_bits = modulator.demodulate(eq_data.flatten())
    
    # 6. Metrics & Visualization
    ber = calculate_ber(tx_bits, rx_bits)
    print(f"Simulation Complete. BER: {ber:.6f} at SNR: {snr_db} dB")
    
    H_true = np.fft.fft(np.pad(H_taps[:, 0], (0, n_subcarriers - H_taps.shape[0])))
    H_true = np.tile(H_true, (n_ofdm_symbols, 1))
    
    # Show plotting pipeline but disable block for tests if needed
    print("Generating explainability visualizations...")
    explainability_pipeline(
        tx_constellation=tx_symbols_reshaped[0],
        rx_constellation=rx_data[0],
        eq_constellation=eq_data[0],
        H_true=H_true,
        H_est=H_est,
        show=False # Set to True to block and view
    )
    print("Run `python main.py` directly and set show=True in visualize.py to see plots interactively.")

if __name__ == "__main__":
    main()
