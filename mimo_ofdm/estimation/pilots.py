import numpy as np

class PilotInserter:
    """
    Comb-type pilot insertion for OFDM.
    """
    def __init__(self, n_subcarriers=64, pilot_spacing=4, pilot_value=1+1j):
        self.n_subcarriers = n_subcarriers
        self.pilot_spacing = pilot_spacing
        self.pilot_indices = np.arange(0, n_subcarriers, pilot_spacing)
        self.data_indices = np.setdiff1d(np.arange(n_subcarriers), self.pilot_indices)
        self.n_pilots = len(self.pilot_indices)
        self.pilot_value = pilot_value

    def insert(self, data_symbols):
        """
        Inserts pilots into the data symbols.
        
        Args:
            data_symbols: Array of shape (n_symbols, len(data_indices))
        Returns:
            ofdm_symbols: Array of shape (n_symbols, n_subcarriers)
        """
        data_symbols = np.asarray(data_symbols)
        if data_symbols.ndim == 1:
            data_symbols = data_symbols.reshape(1, -1)
            
        n_symbols = data_symbols.shape[0]
        ofdm_symbols = np.zeros((n_symbols, self.n_subcarriers), dtype=complex)
        
        # Insert pilots
        for i in self.pilot_indices:
            ofdm_symbols[:, i] = self.pilot_value
            
        # Insert data
        for k, idx in enumerate(self.data_indices):
            ofdm_symbols[:, idx] = data_symbols[:, k]
            
        return ofdm_symbols

    def extract(self, rx_symbols):
        """
        Extracts pilots and data from received OFDM symbols.
        
        Args:
            rx_symbols: Array of shape (n_symbols, n_subcarriers)
        Returns:
            rx_pilots: Array of shape (n_symbols, n_pilots)
            rx_data: Array of shape (n_symbols, len(data_indices))
        """
        rx_symbols = np.asarray(rx_symbols)
        if rx_symbols.ndim == 1:
            rx_symbols = rx_symbols.reshape(1, -1)
            
        rx_pilots = rx_symbols[:, self.pilot_indices]
        rx_data = rx_symbols[:, self.data_indices]
        
        return rx_pilots, rx_data
