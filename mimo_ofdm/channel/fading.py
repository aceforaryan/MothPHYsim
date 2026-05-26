import numpy as np
from scipy.interpolate import interp1d

class TDLChannel:
    """
    Tapped Delay Line (TDL) Rayleigh Fading Channel.
    Models multipath effects and user mobility (Doppler shift).
    """
    def __init__(self, sample_rate, delays, powers_db, max_doppler=0.0, seed=None):
        """
        Args:
            sample_rate: System sampling rate in Hz.
            delays: Array of path delays in seconds.
            powers_db: Array of average path powers in dB.
            max_doppler: Maximum Doppler shift in Hz.
            seed: Random seed.
        """
        self.sample_rate = sample_rate
        self.delays = np.asarray(delays)
        self.powers_linear = 10 ** (np.asarray(powers_db) / 10)
        self.powers_linear /= np.sum(self.powers_linear) # Normalize to 1
        
        self.max_doppler = max_doppler
        self.rng = np.random.default_rng(seed)
        
        self.n_paths = len(self.delays)

    def apply(self, signal):
        """
        Applies TDL fading to the time-domain signal.
        """
        signal = np.asarray(signal)
        n_samples = len(signal)
        
        # Generate fading taps for each path
        if self.max_doppler > 0:
            taps = self._generate_jakes(n_samples)
        else:
            # Static Rayleigh fading (complex Gaussian)
            taps = np.sqrt(0.5) * (self.rng.standard_normal((self.n_paths, 1)) + 
                                   1j * self.rng.standard_normal((self.n_paths, 1)))
            taps = np.tile(taps, (1, n_samples))
            
        # Scale taps by path power
        taps = taps * np.sqrt(self.powers_linear)[:, np.newaxis]
        
        # Apply fading
        delay_samples = np.round(self.delays * self.sample_rate).astype(int)
        
        faded_signal = np.zeros(n_samples, dtype=complex)
        for p in range(self.n_paths):
            d = delay_samples[p]
            if d < n_samples:
                delayed_sig = np.pad(signal, (d, 0))[:n_samples]
                faded_signal += delayed_sig * taps[p, :]
                
        return faded_signal, taps

    def _generate_jakes(self, n_samples, n_oscillators=8):
        """Simplified Jakes fading generator."""
        taps = np.zeros((self.n_paths, n_samples), dtype=complex)
        t = np.arange(n_samples) / self.sample_rate
        
        for p in range(self.n_paths):
            for n in range(n_oscillators):
                alpha = (np.pi * n) / n_oscillators
                f_n = self.max_doppler * np.cos(alpha)
                phase_i = self.rng.uniform(0, 2*np.pi)
                phase_q = self.rng.uniform(0, 2*np.pi)
                
                taps[p, :] += np.cos(2*np.pi*f_n*t + phase_i) + 1j * np.cos(2*np.pi*f_n*t + phase_q)
                
            taps[p, :] /= np.sqrt(n_oscillators)
            
        return taps
