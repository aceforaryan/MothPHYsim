import numpy as np

class OFDM:
    """
    OFDM Modulator / Demodulator (Transmitter / Receiver core operations).
    Handles IFFT/FFT and Cyclic Prefix insertion/removal.
    """
    def __init__(self, n_subcarriers=64, cp_length=16):
        self.n_subcarriers = n_subcarriers
        self.cp_length = cp_length

    def modulate(self, symbols):
        """
        Takes frequency-domain symbols and converts them to time-domain OFDM symbols
        using IFFT, then adds Cyclic Prefix (CP).
        
        Args:
            symbols: Array of shape (n_symbols, n_subcarriers)
        Returns:
            tx_signal: Time domain signal with CP. Shape: (n_symbols, n_subcarriers + cp_length)
        """
        symbols = np.asarray(symbols)
        if symbols.ndim == 1:
            symbols = symbols.reshape(1, -1)
            
        time_domain = np.fft.ifft(symbols, axis=1) * np.sqrt(self.n_subcarriers)
        
        cp = time_domain[:, -self.cp_length:]
        tx_signal = np.concatenate([cp, time_domain], axis=1)
        
        return tx_signal

    def demodulate(self, rx_signal):
        """
        Takes time-domain received signal, removes CP, and converts back to
        frequency-domain symbols using FFT.
        
        Args:
            rx_signal: Time domain signal with CP. Shape: (n_symbols, n_subcarriers + cp_length)
        Returns:
            symbols: Frequency domain symbols. Shape: (n_symbols, n_subcarriers)
        """
        rx_signal = np.asarray(rx_signal)
        if rx_signal.ndim == 1:
            rx_signal = rx_signal.reshape(1, -1)
            
        time_domain = rx_signal[:, self.cp_length:]
        symbols = np.fft.fft(time_domain, axis=1) / np.sqrt(self.n_subcarriers)
        
        return symbols
