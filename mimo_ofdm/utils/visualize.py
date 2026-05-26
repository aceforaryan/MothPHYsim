import matplotlib.pyplot as plt
import numpy as np

def plot_constellation(symbols, title="Constellation Diagram"):
    """
    Plots a constellation diagram.
    """
    plt.figure(figsize=(6, 6))
    plt.scatter(np.real(symbols), np.imag(symbols), alpha=0.5, color='b')
    plt.axhline(0, color='black', lw=1)
    plt.axvline(0, color='black', lw=1)
    plt.title(title)
    plt.xlabel("In-Phase")
    plt.ylabel("Quadrature")
    plt.grid(True)
    # plt.show() # Disabled for automated tests, uncomment when running manually

def plot_channel_response(H_true, H_est=None, title="Channel Frequency Response"):
    """
    Plots the true and estimated channel magnitude responses.
    """
    plt.figure(figsize=(10, 4))
    
    if H_true.ndim > 1:
        H_true = H_true[0]
        
    plt.plot(np.abs(H_true), label="True Channel", lw=2)
    
    if H_est is not None:
        if H_est.ndim > 1:
            H_est = H_est[0]
        plt.plot(np.abs(H_est), '--', label="Estimated Channel", lw=2)
        
    plt.title(title)
    plt.xlabel("Subcarrier Index")
    plt.ylabel("Magnitude")
    plt.legend()
    plt.grid(True)
    # plt.show()
    
def explainability_pipeline(tx_constellation, rx_constellation, eq_constellation, H_true, H_est, show=True):
    """
    Displays the full pipeline: Bits -> Constellation -> Channel -> Equalized
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # TX Constellation
    axes[0, 0].scatter(np.real(tx_constellation), np.imag(tx_constellation), alpha=0.5)
    axes[0, 0].set_title("Transmitted Constellation")
    axes[0, 0].grid(True)
    
    # RX Constellation (before equalization)
    axes[0, 1].scatter(np.real(rx_constellation), np.imag(rx_constellation), alpha=0.5, color='r')
    axes[0, 1].set_title("Received Constellation (Faded & Noisy)")
    axes[0, 1].grid(True)
    
    # Equalized Constellation
    axes[1, 0].scatter(np.real(eq_constellation), np.imag(eq_constellation), alpha=0.5, color='g')
    axes[1, 0].set_title("Equalized Constellation")
    axes[1, 0].grid(True)
    
    # Channel Response
    H_true_plt = H_true[0] if H_true.ndim > 1 else H_true
    H_est_plt = H_est[0] if H_est.ndim > 1 else H_est
    axes[1, 1].plot(np.abs(H_true_plt), label="True", lw=2)
    axes[1, 1].plot(np.abs(H_est_plt), '--', label="Estimated", lw=2)
    axes[1, 1].set_title("Channel Magnitude Response")
    axes[1, 1].legend()
    axes[1, 1].grid(True)
    
    plt.tight_layout()
    if show:
        plt.show()
    return fig
