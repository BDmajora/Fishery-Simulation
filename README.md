# Survey-Based Fishery Management Simulation

A high-performance Management Strategy Evaluation (MSE) tool designed for data-limited fisheries. This project simulates fish population dynamics and evaluates target-based management procedures to prevent stock collapse and optimize fishing effort.

## Overview

Based on research by Sun et al. (2020), this simulation implements Target-Based Management Procedures. It compares current abundance indices against fixed target levels to dynamically adjust fishing pressure, providing a computational framework for fisheries lacking traditional stock assessment data.

## Features

* **Population Dynamics Modeling:** Simulates fish stock abundance, recruitment, and mortality using NumPy and Pandas.
* **Target-Based Feedback Loops:** Automated management logic that scales fishing effort based on real-time abundance deviations.
* **Hardware-Accelerated Simulation:** Optional PyTorch GPU acceleration to scale simulations to larger datasets and complex multi-species environments.
* **Performance Benchmarking:** Comparative analysis tools to measure execution latency and throughput differences between CPU (NumPy) and GPU (PyTorch) backends.
* **Data Visualization:** Integrated Matplotlib suite for tracking population recovery trends and collapse probabilities.

## Dependencies

This project is built with Python 3.10+ and utilizes the following libraries:

* **NumPy**: Core numerical routines for population matrix manipulations and linear algebra.
* **Pandas**: Data structures for tracking historical simulation logs and fishery metrics.
* **Matplotlib**: Generative plotting for management strategy evaluation and trend analysis.
* **PyTorch**: GPU-accelerated tensor operations for high-throughput simulation runs.

## Installation

1. **Clone the repository:**
   git clone https://github.com/your-username/fishery-simulation.git
   cd fishery-simulation

2. **Set up a virtual environment:**
   python3 -m venv venv
   source venv/bin/activate

3. **Install dependencies:**
   pip install -r requirements.txt

## Usage

Run with the run.py in the root directory:

python3 run.py

## Bibliography

Sun, M., Li, Y., Ren, Y., & Chen, Y. (2020). Using fisheries-independent survey data to reinforce China’s data-limited fisheries management: Management strategy evaluation of survey-based management procedures. Fisheries Management and Ecology, 27(6), 543–556. https://doi.org/10.1111/fme.12454