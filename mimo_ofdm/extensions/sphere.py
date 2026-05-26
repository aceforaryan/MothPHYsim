import numpy as np

class SphereDecoder:
    """
    Sphere Decoder for MIMO Systems.
    Provides Maximum Likelihood (ML) performance with reduced complexity compared to exhaustive search.
    """
    def __init__(self, constellation, radius=10.0):
        self.constellation = constellation
        self.radius = radius

    def decode(self, y, H):
        """
        Stub for Sphere Decoding algorithm.
        Falls back to Zero Forcing for the stub version.
        """
        epsilon = 1e-10
        H_inv = np.linalg.pinv(H + epsilon * np.eye(H.shape[0]))
        x_est = np.dot(H_inv, y)
        
        distances = np.abs(x_est[:, np.newaxis] - self.constellation[np.newaxis, :])
        indices = np.argmin(distances, axis=1)
        return self.constellation[indices]
