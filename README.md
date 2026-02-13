# QAOA-DME TSP Benchmark 

## Overview
This repository benchmarks a DME-inspired QAOA variant for the Traveling Salesman Problem (TSP). It encodes only valid tours into a Hilbert space and studies how circuit depth affects solution quality under a fixed, linear parameter schedule for gamma and beta. Two notebooks define the full pipeline: one for dataset generation and one for benchmarking.


## Method Summary
### TSP Encoding
- Cities are labeled with a fixed start/end at city 0.
- All valid permutations of the remaining cities are mapped to computational basis states.
- A uniform superposition over valid tours is prepared (state |F>).

### Quantum Circuit
- Alternating operators implement a QAOA-like evolution:
  - Cost oracle: phase shifts using tour lengths.
  - Mixer: Grover-style reflection about |F>.
- Depth p is varied over a log-scaled schedule (e.g., 2, 4, 8, 16, 32).

### Error Metric
- For each instance, costs are min-max normalized between optimal and worst-case tour lengths.
- The reported error e_L is the expected normalized cost from the circuit output distribution.

## Dataset Generation
200 Instances per dataset are synthesized with two distributions:
- 150 Instances with perturbed circular placements (multiple noise levels).
- 50 Instances with uniform random placements in the unit square.

A greedy filter removes instances solvable by nearest-neighbor heuristics from any start city. Exact optimal and worst-case tour lengths are computed via dynamic programming.

## Outputs
Benchmark outputs are saved per N in `Benchmark_Results/`:
- Raw instance-level errors.
- Mean error by depth.
- Environment metadata.
- Compressed arrays and plots.

Generated datasets are stored in `Generated_datasets/` as `.npz` archives with coordinates, distance matrices, and extrema.

## Repository Layout
- `DME_tsp_benchmark_final_without_param_optimization.ipynb`: benchmarking notebook.
- `generate_dataset.ipynb`: dataset synthesis notebook.
- `Benchmark_Results/`: saved results and plots by problem size.
- `Generated_datasets/`: saved instance datasets.
- `requirements_benchmark.txt`: environment snapshot for benchmarking.
- `requirements_generate_dataset.txt`: environment snapshot for dataset generation.


## Reproducibility
The notebooks capture system metadata and freeze dependencies. To replicate results, use the same Python version and install the recorded requirements.

### 1. Environment Setup
Create and activate a virtual environment, then install dependencies:

```powershell
# Create virtual environment
python -m venv .example_env

# Activate the environment
.\.example_env\Scripts\Activate.ps1  # For PowerShell
.dme_quaoa_tsp\Scripts\activate  # For Windows cmd
source .dme_quaoa_tsp/bin/activa  # For macOS/Linux

# Install dependencies
pip install -r requirements_generate_dataset.txt
pip install -r requirements_benchmark.txt
```

#### Requirements Snapshots
- Dataset generation environment: `requirements_generate_dataset.txt`.
- Benchmarking environment: `requirements_benchmark.txt`.

#### Environment Metadata
- Dataset generation metadata is stored inside each `.npz` in `Generated_datasets/` under the `metadata` key.
- Benchmarking metadata is saved per run in `Benchmark_Results/N_*_Cities/*_metadata.json`.

### 2. Generate Datasets
Open `generate_dataset.ipynb`, specify the desired number of cities and seed and run all cells. By default, it generates data for a specific N and seed and writes to `Generated_datasets/`.

### 3. Run Benchmarks
Open `DME_tsp_benchmark_final_without_param_optimization.ipynb`, specify the desired number of cities and seed and run all cells. It loads a dataset from `Generated_datasets/`, evaluates multiple depths, and saves results in `Benchmark_Results/`.

