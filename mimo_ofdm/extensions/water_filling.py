import numpy as np

def water_filling(channel_gains, total_power, noise_variance):
    """
    Allocates power across subcarriers to maximize capacity based on the water-filling algorithm.
    """
    gains = np.asarray(channel_gains)
    n_channels = len(gains)
    
    # Inverse SNR for each channel
    inv_snr = noise_variance / (np.abs(gains)**2 + 1e-10)
    
    sorted_indices = np.argsort(inv_snr)
    sorted_inv_snr = inv_snr[sorted_indices]
    
    power_allocation = np.zeros(n_channels)
    
    for i in range(n_channels, 0, -1):
        water_level = (total_power + np.sum(sorted_inv_snr[:i])) / i
        
        if water_level >= sorted_inv_snr[i-1]:
            p_alloc = water_level - sorted_inv_snr[:i]
            power_allocation[sorted_indices[:i]] = p_alloc
            break
            
    return power_allocation
