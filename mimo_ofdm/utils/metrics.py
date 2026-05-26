import numpy as np

def calculate_ber(tx_bits, rx_bits):
    """
    Calculates the Bit Error Rate (BER).
    """
    tx_bits = np.asarray(tx_bits)
    rx_bits = np.asarray(rx_bits)
    
    min_len = min(len(tx_bits), len(rx_bits))
    errors = np.sum(tx_bits[:min_len] != rx_bits[:min_len])
    
    return errors / min_len

def calculate_throughput(n_bits, duration_seconds):
    """
    Calculates throughput in bits per second.
    """
    return n_bits / duration_seconds

def calculate_evm(tx_symbols, rx_symbols):
    """
    Calculates Error Vector Magnitude (EVM) in percentage.
    """
    tx_symbols = np.asarray(tx_symbols)
    rx_symbols = np.asarray(rx_symbols)
    
    min_len = min(len(tx_symbols), len(rx_symbols))
    error_vector = tx_symbols[:min_len] - rx_symbols[:min_len]
    
    evm_rms = np.sqrt(np.mean(np.abs(error_vector)**2))
    ref_rms = np.sqrt(np.mean(np.abs(tx_symbols[:min_len])**2))
    
    return (evm_rms / ref_rms) * 100.0
