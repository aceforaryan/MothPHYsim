import numpy as np

class SphereDecoder:
    """
    Sphere Decoder for MIMO Systems.
    Provides Maximum Likelihood (ML) performance with reduced complexity compared to exhaustive search.
    """
    def __init__(self, constellation, radius=10.0):
        self.constellation = np.asarray(constellation, dtype=complex)
        self.radius = radius

    def decode(self, y, H):
        """
        Performs Sphere Decoding (Maximum Likelihood detection) for y = H*x + n.
        
        Args:
            y: Received vector of shape (n_rx,)
            H: Channel matrix of shape (n_rx, n_tx)
        Returns:
            x_best: Best complex constellation symbol vector of shape (n_tx,)
        """
        y = np.asarray(y, dtype=complex)
        H = np.asarray(H, dtype=complex)
        
        n_rx, n_tx = H.shape
        
        # QR decomposition of channel matrix H
        # H = Q * R, where Q is unitary (n_rx, n_tx) and R is upper triangular (n_tx, n_tx)
        Q, R = np.linalg.qr(H)
        
        # Transform the received vector
        z = np.dot(Q.conj().T, y) # (n_tx,)
        
        best_x = None
        best_dist = float('inf')
        
        # Current candidate path
        current_x = np.zeros(n_tx, dtype=complex)
        
        def search(level, accumulated_dist):
            nonlocal best_dist, best_x
            
            # Prune if accumulated distance is already worse than the best distance
            if accumulated_dist >= best_dist:
                return
                
            if level < 0:
                # We have reached a leaf node, update the best candidate
                best_dist = accumulated_dist
                best_x = current_x.copy()
                return
            
            # Calculate the unconstrained estimate at the current level
            # z_level = R[level, level] * x_level + sum_{j=level+1}^{n_tx-1} R[level, j] * x_j
            interference = 0.0
            for j in range(level + 1, n_tx):
                interference += R[level, j] * current_x[j]
                
            hat_x = (z[level] - interference) / R[level, level]
            
            # Schnorr-Euchner ordering: Sort constellation points by distance to hat_x
            distances_to_hat = np.abs(self.constellation - hat_x)**2
            sorted_indices = np.argsort(distances_to_hat)
            sorted_constellation = self.constellation[sorted_indices]
            
            for c in sorted_constellation:
                dist_contribution = np.abs(z[level] - R[level, level] * c - interference)**2
                new_dist = accumulated_dist + dist_contribution
                
                # Check sphere boundary constraint
                if new_dist < best_dist:
                    current_x[level] = c
                    search(level - 1, new_dist)
                    
        # Start search from the last level (n_tx - 1) down to 0
        search(n_tx - 1, 0.0)
        
        if best_x is None:
            # Fallback to Zero Forcing if no node was found within the radius (highly unlikely with best_dist=inf)
            epsilon = 1e-10
            H_inv = np.linalg.pinv(H + epsilon * np.eye(H.shape[0]))
            x_est = np.dot(H_inv, y)
            distances = np.abs(x_est[:, np.newaxis] - self.constellation[np.newaxis, :])
            indices = np.argmin(distances, axis=1)
            best_x = self.constellation[indices]
            
        return best_x
