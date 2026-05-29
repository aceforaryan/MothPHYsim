import numpy as np


class AWGNChannel:
    """
    Additive White Gaussian Noise (AWGN) Channel.

    SNR definition — Es/N0
    -----------------------
    The ``snr_db`` parameter is the **symbol energy to noise ratio Es/N0**,
    defined as:

        SNR = mean(|signal|²) / mean(|noise|²)

    This equals Es/N0 for unit-power constellations (which QAMModulator
    guarantees by normalising constellation points to mean power 1).

    Relationship to Eb/N0 (bit energy to noise ratio):
        Eb/N0 (dB) = Es/N0 (dB) − 10 · log10(k)
    where k = log2(M) bits per symbol.  Concretely:
        QPSK (k=2):   Es/N0 = Eb/N0 + 3.01 dB
        16-QAM (k=4): Es/N0 = Eb/N0 + 6.02 dB

    All BER-vs-SNR plots in this project use Es/N0 on the x-axis.
    To compare against textbook Eb/N0 curves, shift the axis by −k dB.
    """

    def __init__(self, seed=None):
        self.rng = np.random.default_rng(seed)

    def apply(self, signal, snr_db):
        """
        Adds AWGN to the signal based on the target Es/N0.

        Args:
            signal: Transmitted signal (complex or real array).
            snr_db: Target Es/N0 in dB.

        Returns:
            noisy_signal: Signal with added AWGN.
        """
        signal = np.asarray(signal)

        # Calculate signal power
        sig_power = np.mean(np.abs(signal) ** 2)

        # Calculate required noise power from Es/N0
        snr_linear = 10 ** (snr_db / 10)
        noise_power = sig_power / snr_linear

        # Generate complex Gaussian noise.
        # Divide noise power equally between real and imaginary parts.
        noise = np.sqrt(noise_power / 2) * (
            self.rng.standard_normal(signal.shape)
            + 1j * self.rng.standard_normal(signal.shape)
        )

        return signal + noise
