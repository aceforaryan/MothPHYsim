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
- `mimo_ofdm/extensions/`: Advanced V2 features like Sphere decoding and Water Filling
- `experiments/`: Harness for comparing estimators, modulation, and MIMO schemes

## Quick Start
Install dependencies and sync the virtual environment using `uv`:
```bash
uv sync
```

Alternatively, you can install the dependencies via pip:
```bash
pip install -r requirements.txt
```

### Running the System
If you are using `uv`, you can prefix execution with `uv run`. Otherwise, ensure your virtual environment is activated.

Run the core simulation pipeline (generates explainability plots):
```bash
uv run python main.py
# or if environment is activated: python main.py
```

*Note: In `main.py`, set `show=True` in the `explainability_pipeline()` call to interactively view the constellation, pilot maps, and equalized bits!*

## Running Experiments
The project includes an experiment harness to evaluate various system parameters over an SNR range.

1. **Compare Estimators (LS vs MMSE):**
```bash
uv run python experiments/compare_estimators.py
```

2. **Compare Modulation (QPSK vs 16-QAM):**
```bash
uv run python experiments/compare_modulation.py
```

3. **Compare MIMO Configurations (2x2 vs 4x4) (Stub):**
```bash
uv run python experiments/compare_mimo.py
```

## Running Tests
Run the comprehensive `pytest` suite:
```bash
uv run pytest tests/
```

