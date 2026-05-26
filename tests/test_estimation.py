import numpy as np
from mimo_ofdm.estimation.pilots import PilotInserter
from mimo_ofdm.estimation.ls import LSEstimator

def test_pilot_insertion_extraction():
    n_subcarriers = 64
    pilots = PilotInserter(n_subcarriers, pilot_spacing=4)
    
    data = np.ones((1, len(pilots.data_indices)))
    ofdm_sym = pilots.insert(data)
    
    p_ext, d_ext = pilots.extract(ofdm_sym)
    
    assert np.allclose(d_ext, data)
    assert np.allclose(p_ext, pilots.pilot_value)
    
def test_ls_estimator():
    n_subcarriers = 64
    pilots = PilotInserter(n_subcarriers, pilot_spacing=4)
    estimator = LSEstimator(pilots.pilot_indices, n_subcarriers)
    
    # Simulate a flat channel H = 2
    rx_pilots = np.ones((1, len(pilots.pilot_indices))) * pilots.pilot_value * 2
    H_est = estimator.estimate(rx_pilots)
    
    assert np.allclose(H_est, 2)
