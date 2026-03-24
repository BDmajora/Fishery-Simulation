# High-Throughput Tensor-Based Fishery Management Simulation

A performance-optimized Management Strategy Evaluation (MSE) framework engineered to profile and accelerate biological population dynamics through hardware-accelerated tensor mathematics. This project demonstrates a transition from legacy sequential modeling to a high-parallelism GPU execution environment using PyTorch.

## Engineering Highlights

* **Tensor-Driven Dynamics:** Migrated core simulation logic from NumPy to **PyTorch**, utilizing vectorized operations to eliminate Python-level bottleneck loops and maximize arithmetic intensity.
* **Hardware-Agnostic Acceleration:** Optimized for target hardware by implementing dynamic device allocation (`CUDA`/`CPU`), ensuring efficient memory management and minimizing PCIe overhead.
* **Performance Profiling & Benchmarking:** Integrated a comparative analysis suite to measure **Latency (ms/step)** and **Throughput (steps/sec)**, documenting performance gains achieved via GPU-accelerated tensor contractions.
* **Vectorized State Management:** Implemented efficient tensor slicing, broadcasting, and dimension-based reductions (`torch.sum`) to manage multi-patch state transitions without iterative overhead.
* **Stochastic Modeling:** Applied probability theory and noise injection directly on the GPU/TPU device to maintain high-throughput execution during Monte Carlo-style simulation runs.

## Technical Competencies Demonstrated

| Competency | Implementation Detail |
| :--- | :--- |
| **Tensor Math** | Leveraged broadcasting, outer products, and dimension-specific reductions to simulate complex interactions. |
| **Performance Optimization** | Refactored $O(N)$ Python loops into $O(1)$ vectorized operations for spatial patch dynamics. |
| **Hardware Utilization** | Profiled end-to-end performance differences between CPU (NumPy) and GPU (PyTorch) backends. |
| **Systems Architecture** | Developed a modular, object-oriented simulation engine with clean separation of physics and management logic. |

## Dependencies

* **PyTorch**: High-performance tensor operations and GPU acceleration logic.
* **NumPy**: Base numerical routines and legacy data comparison.
* **Pandas**: Efficient logging and structured data manipulation for simulation history.
* **Matplotlib**: Generative plotting for analyzing population recovery trends and management efficacy.

## Installation

1. **Clone the repository:**
   git clone https://github.com/bdmajora/fishery-simulation.git
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
