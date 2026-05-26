import numpy as np

class AWGNChannel:
    """
    Additive White Gaussian Noise (AWGN) Channel.
    """
    def __init__(self, seed=None):
        self.rng = np.random.default_rng(seed)

    def apply(self, signal, snr_db):
        """
        Adds AWGN to the signal based on the target SNR (Signal-to-Noise Ratio).
        
        Args:
            signal: Transmitted signal (complex or real array).
            snr_db: Target Signal-to-Noise Ratio in dB.
            
        Returns:
            noisy_signal: Signal with added AWGN.
        """
        signal = np.asarray(signal)
        
        # Calculate signal power
        sig_power = np.mean(np.abs(signal)**2)
        
        # Calculate required noise power
        snr_linear = 10 ** (snr_db / 10)
        noise_power = sig_power / snr_linear
        
        # Generate complex Gaussian noise
        # Note: Divide noise power by 2 for real and imaginary parts
        noise = np.sqrt(noise_power / 2) * (self.rng.standard_normal(signal.shape) + 
                                            1j * self.rng.standard_normal(signal.shape))
                                            
        return signal + noise
