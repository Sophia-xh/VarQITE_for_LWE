# VarQITE for LWE

Reproducibility repository for the paper  
*“Variational quantum imaginary-time evolution for the learning with errors problem”*.

## Overview

This work proposes a Babai-guided local variational quantum imaginary-time
evolution (VarQITE) scheme for search-LWE secret recovery.

An LWE instance is reformulated as a q-ary lattice decoding problem.
Babai's nearest-plane algorithm first identifies a local region, VarQITE
generates low-energy correction candidates within that region, and an
energy-guided LWE-prior reranking rule selects the final recovered secret.

This repository contains the implementation, experimental data, and scripts
used to reproduce Figures 2–4 of the paper. All quantum circuits are simulated
with the PyQPanda3 CPUQVM simulator.

## Repository Structure

```text
.
├── algorithms/              Core algorithm implementations
├── scripts/
│   ├── calibration/         Sample-ratio calibration and Figure 2
│   ├── performance/         Main experiments, baselines, and Figure 3
│   └── case_study/          Representative-instance analysis and Figure 4
├── data/
│   ├── figure2_sample_ratio_calibration/
│   ├── figure3_performance_comparison/
│   └── figure4_mechanism_analysis/
├── figures/                 Figures used in the paper
├── experiment_settings.json
├── requirements.txt
└── LICENSE
```

`experiment_settings.json` summarizes the experimental configuration.

## Installation

Tested with Python 3.10.20.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Main dependencies:

```text
matplotlib==3.10.8
numpy==2.2.6
pandas==2.3.3
pyqpanda3==0.3.4
scipy==1.15.3
sympy==1.14.0
```

## Reproducing the Paper Figures

All commands are run from the repository root.

### Figure 2 — Sample-ratio calibration

```bash
python scripts/calibration/plot_figure2_sample_ratio_calibration.py
```

Data: `data/figure2_sample_ratio_calibration/`

### Figure 3 — Performance comparison

```bash
python scripts/performance/plot_figure3_performance_comparison.py
```

Data: `data/figure3_performance_comparison/`

### Figure 4 — Mechanism analysis

```bash
python scripts/case_study/plot_figure4_mechanism_analysis.py
```

Data: `data/figure4_mechanism_analysis/`

The plotting command uses the archived experimental data and does not rerun
VarQITE. The original shot-measured energy surface in Figure 4(b) was not archived and is therefore omitted from the simulation-free replot.

## Re-running Experiments

The main VarQITE experiment can be rerun with:

```bash
python scripts/performance/run_main_varqite_experiment.py
```

The sample-ratio calibration can be rerun with:

```bash
python scripts/calibration/run_sample_ratio_calibration.py
```

The representative-instance case study can be rerun with:

```bash
python scripts/case_study/run_figure4_case_study.py
```

Re-running these experiments is not required to reproduce the archived paper
figures.

## Experimental Settings

The main reported setting uses:

- `q = 17`
- `omega = 0.5`
- `tau = 0.25`
- `c* = 1.325`
- local correction window `{-1, 0, 1}`
- one-layer `R_y` product ansatz
- `dtau = 0.02`
- `T = 5` VarQITE updates
- top-`K = 20` candidates

The complete configuration is available in `experiment_settings.json`.

## License

Released under the MIT License. See `LICENSE`.

## Citation

If you use this repository, please cite the associated paper and archived
release. Citation information will be updated once available.
