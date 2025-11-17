This repository contains the data, code, and documentation supporting the study:

Yang, J. et al., 2025. **"Identifying Dominant Parameters Across Space and Time at Multiple Scales in a Distributed Model Using a Two-Step Deep Learning-Assisted Time-Varying Spatial Sensitivity Analysis"** (submitted for publication in HESS).

## 🧭 Overview

This study proposes a two-step deep learning-assisted time-varying spatial sensitivity analysis (SSA) framework for distributed hydrological models. Using the Soil and Water Assessment Tool (SWAT) for the Jinghe River Basin, we first use a global screening step to identify influential parameters, and then apply deep learning-assisted Sobol analysis to quantify how parameter importance evolves across space (subbasins and hydrological response units) and time (rolling windows). The framework reveals scale-dependent sensitivity hotspots and hot moments that guide targeted calibration and more reliable water management.

## 📁 Repository Structure
```bash
📂 TwoStep_DL_TimeVarying_SSA/
├── Figure1/                      # Data and scripts used to generate Figure 1
├── Figure2/                      # Data and scripts used to generate Figure 2
├── Figure3/                      # Data and scripts used to generate Figure 3
├── Figure4/                      # Data and scripts used to generate Figure 4
├── Figure5/                      # Data and scripts used to generate Figure 5
├── Figure6/                      # Data and scripts used to generate Figure 6
├── Figure7/                      # Data and scripts used to generate Figure 7
└── lib/                          # Shared functions and modules used across figures
```

A typical FigureX/ folder contains:

- Input data required for that figure (parameter sets, model outputs, sensitivity indices, etc.)

- Analysis scripts / notebooks to compute indices and generate plots

- Example outputs (figures or intermediate result files)

The lib/ directory contains reusable modules, for example: Utility functions for plotting and post-processing.

## 🛠 Requirements

The code primarily uses Python; some steps may also use SWAT or MATLAB/R depending on your setup.

Python (recommended ≥ 3.9)
Typical packages include:

numpy

pandas

matplotlib

scipy

SALib (for global sensitivity analysis)

torch (PyTorch, for deep learning surrogates)


## 📌 Key Features

- Two-step framework combining global screening and deep learning-assisted Sobol analysis

- Multi-scale parameterization at subbasin and hydrological response unit levels

- Time-varying sensitivity indices using rolling windows to capture nonstationary behavior

- Identification of sensitivity hotspots and hot moments, supporting targeted calibration and monitoring design

- Fully organized repository enabling figure-by-figure reproducibility

## 🌍 Broader Applications

The framework and code structure can be adapted to:

Other SWAT applications or distributed hydrological models

Spatiotemporal sensitivity analysis in ecohydrology, water quality, or land–atmosphere studies

Any model where parameter importance varies across both space and time and computational cost is high.

## 📜 Citation

If you use this repository, please cite:

Yang, J. et al., 2025. Identifying Dominant Parameters Across Space and Time at Multiple Scales in a Distributed Model Using a Two-Step Deep Learning-Assisted Time-Varying Spatial Sensitivity Analysis. (in review, HESS).

## 📬 Contact

For questions, please contact:  
**Jing Yang**  
School of Land Engineering, Chang’an University 

Email: jing.yang@126.com
