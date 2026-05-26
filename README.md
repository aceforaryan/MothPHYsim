# MIMO-OFDM Wireless Communication System

This is a comprehensive, highly-modular simulator for a MIMO-OFDM wireless communication system featuring adaptive channel estimation.

## Features
- **Core OFDM & Modulation:** QPSK/16QAM, IFFT/FFT processing with Cyclic Prefix.
- **Wireless Channels:** TDL Rayleigh fading, multipath reflections, and AWGN.
- **MIMO:** 2x2 and 4x4 spatial multiplexing.
- **Estimation & Equalization:** Comb pilot extraction, LS/MMSE estimation, and ZF/MMSE equalizers.
- **Explainability:** Step-by-step visualizations from bits to recovered symbols using matplotlib.

## Structure
- `mimo_ofdm/core/`: Modulation, OFDM, MIMO
- `mimo_ofdm/channel/`: TDL Fading, AWGN
- `mimo_ofdm/estimation/`: Pilot insertion, LS/MMSE Channel Estimation
- `mimo_ofdm/equalization/`: ZF/MMSE Equalization
- `mimo_ofdm/extensions/`: Advanced V2 features like Sphere decoding
- `experiments/`: Harness for comparing estimators, modulation, and MIMO schemes

## Quick Start
```bash
uv pip install -r requirements.txt
python main.py
```
