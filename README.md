
# Quant-Algorithms(CTAEA): Paper implementation and improvements

This repository provides both the full source code and the complete research workflow for constrained multi-objective optimization experiments. The folders `CTAEA-4` and `CTAEA-Curr` contain implementations for all six algorithms studied (not just CTAEA), enabling direct experimentation and extension (Curr represent the current version on which we are working and 4 represent the last version which was fully implemented) (other versions can be found in the git history, as deleted).

In addition to the code, the repo includes experiment outputs, plots, paper artifacts, and scripts for aggregating and presenting results.

The central question addressed here is: how do six state-of-the-art constrained multi-objective algorithms perform across a suite of benchmark problems with varying numbers of objectives ($m \in \{3,5,8,10,15\}$)?


## CTAEA: Main Paper
This project is primarily based on the paper:

**A Two-Archive Evolutionary Algorithm for Constrained Multi-Objective Optimization**

- **Authors:** Y. Wang, B. Li, X. Yao
- **Published in:** IEEE Transactions on Evolutionary Computation, 2020
- **Direct PDF:** [Papers/Main-Paper.pdf](Papers/Main-Paper.pdf)
- **IEEE Xplore:** [A Two-Archive Evolutionary Algorithm for Constrained Multi-Objective Optimization](https://ieeexplore.ieee.org/document/9099352)

### Summary
CTAEA (Constrained Two-Archive Evolutionary Algorithm) is a state-of-the-art method for solving constrained multi-objective optimization problems (CMOPs). The key innovation is the use of two co-evolving archives:

- **Feasible Archive (FA or CA):** Focuses on maintaining and improving feasible solutions, driving convergence to the true Pareto front.
- **Infeasible Archive (IA or DA):** Maintains high-quality infeasible solutions, which helps the search escape local optima and explore the boundary between feasible and infeasible regions.

The algorithm alternates between updating these two archives, using a tailored selection and variation strategy. This dual-archive approach allows CTAEA to efficiently balance feasibility and optimality, outperforming many previous algorithms on a wide range of benchmark problems.

**Main contributions:**
- Proposes a two-archive framework for CMOPs
- Demonstrates superior performance on both classic and difficult constrained benchmarks
- Provides a robust mechanism for handling disconnected or complex feasible regions

For full details, see the [Main-Paper.pdf](Papers/Main-Paper.pdf) in the Papers directory or the [official IEEE Xplore page](https://ieeexplore.ieee.org/document/9099352).


## Models or Algorithms
There are 6 models/algorithms used in the work. Five are state-of-the-art (SOTA) and one is a new test benchmark. Their original papers are linked below:

- **IDBEA** (SOTA): [Improved Decomposition-Based Evolutionary Algorithm for Constrained Multiobjective Optimization](https://ieeexplore.ieee.org/document/7969316)
- **NSGA-III** (SOTA): [An Evolutionary Many-Objective Optimization Algorithm Using Reference-Point-Based Nondominated Sorting Approach, Part I: Solving Problems With Box Constraints](https://ieeexplore.ieee.org/document/6600851)
- **CMOEA** (SOTA): [Constrained Multiobjective Evolutionary Algorithm With Dominance and Decomposition](https://ieeexplore.ieee.org/document/7553459)
- **CMOEA/D** (SOTA): [Constrained Multiobjective Evolutionary Algorithm Based on Decomposition](https://ieeexplore.ieee.org/document/7553458)
- **CMOEA/DD** (SOTA): [Constrained Multiobjective Evolutionary Algorithm With Dual Decomposition](https://ieeexplore.ieee.org/document/7553457)
- **CTAEA** (new test marker): [A Two-Archive Evolutionary Algorithm for Constrained Multi-Objective Optimization](https://ieeexplore.ieee.org/document/9099352)

You can find more details in the "Papers/" directory or by following the above links to the official IEEE Xplore pages.


## Problems
The simulations are conducted on 11 benchmark problems, divided into two main classes:

- **C-type**: Classic constrained DTLZ problems, where constraints are added to the original DTLZ suite to test the ability of algorithms to handle feasible/infeasible regions.
- **DC-type**: Difficult constrained DTLZ problems, designed to be more challenging by introducing disconnected or complex feasible regions.

Each problem is a variant of the DTLZ family, widely used in multi-objective optimization research. Below is a brief explanation and a reference for each:

- **C1DTLZ1**: Constrained version of DTLZ1, with linear constraints. Tests the ability to find solutions on a linear Pareto front under constraints. [DTLZ1 original paper](https://ieeexplore.ieee.org/document/1290934)
- **C1DTLZ3**: Constrained DTLZ3, with a spherical Pareto front and additional constraints. [DTLZ3 original paper](https://ieeexplore.ieee.org/document/1290934)
- **C2DTLZ2**: Constrained DTLZ2, with a spherical Pareto front and constraints that create disconnected feasible regions. [DTLZ2 original paper](https://ieeexplore.ieee.org/document/1290934)
- **C3DTLZ1**: Another constrained DTLZ1 variant, with different constraint formulations to increase difficulty. [DTLZ1 original paper](https://ieeexplore.ieee.org/document/1290934)
- **C3DTLZ4**: Constrained DTLZ4, introduces a biased mapping to the Pareto front, making the search space more challenging. [DTLZ4 original paper](https://ieeexplore.ieee.org/document/1290934)
- **DC1DTLZ1**: "Difficult" constrained DTLZ1, with constraints that create disconnected feasible regions. [DC-DTLZ reference](https://ieeexplore.ieee.org/document/7553457)
- **DC1DTLZ3**: Difficult constrained DTLZ3, with complex constraints and a spherical front. [DC-DTLZ reference](https://ieeexplore.ieee.org/document/7553457)
- **DC2DTLZ1**: Difficult constrained DTLZ1, with a different set of constraints to further increase problem complexity. [DC-DTLZ reference](https://ieeexplore.ieee.org/document/7553457)
- **DC2DTLZ3**: Difficult constrained DTLZ3, with highly disconnected feasible regions. [DC-DTLZ reference](https://ieeexplore.ieee.org/document/7553457)
- **DC3DTLZ1**: Difficult constrained DTLZ1, with even more challenging constraint structure. [DC-DTLZ reference](https://ieeexplore.ieee.org/document/7553457)
- **DC3DTLZ3**: Difficult constrained DTLZ3, the most complex in the suite, with multiple disconnected feasible regions. [DC-DTLZ reference](https://ieeexplore.ieee.org/document/7553457)


### Table of Reference Points

| Problem Type | PF Type | Reference Point for HV |
|--------------|---------|------------------------|
| C1-DTLZ1 | Linear (sum=0.5) | (1.1, ..., 1.1) |
| C1-DTLZ3 | Sphere quadrant | (1.1, ..., 1.1) |
| C2-DTLZ2 | Sphere quadrant | (1.1, ..., 1.1) |
| C3-DTLZ1 | Linear (sum=0.5) | (1.1, ..., 1.1) |
| C3-DTLZ4 | Sphere quadrant | (2.1, ..., 2.1) |
| DC1-DTLZ1 | Linear | (1.1, ..., 1.1) |
| DC1-DTLZ3 | Sphere | (1.1, ..., 1.1) |
| DC2-DTLZ1 | Linear | (1.1, ..., 1.1) |
| DC2-DTLZ3 | Sphere | (1.1, ..., 1.1) |
| DC3-DTLZ1 | Linear | (1.1, ..., 1.1) |
| DC3-DTLZ3 | Sphere | (1.1, ..., 1.1) |


**References:**
- K. Deb, L. Thiele, M. Laumanns, and E. Zitzler, "Scalable Test Problems for Evolutionary Multiobjective Optimization," Evolutionary Multiobjective Optimization, 2005. ([DTLZ original paper](https://ieeexplore.ieee.org/document/1290934))
- Y. Tian, X. Zhang, C. Zhang, and Y. Jin, "A Multiobjective Evolutionary Algorithm Based on Dominance and Decomposition for Multiobjective Optimization Problems with Complicated Pareto Sets," IEEE Transactions on Evolutionary Computation, 2016. ([DC-DTLZ reference](https://ieeexplore.ieee.org/document/7553457))



## Problem Variations
Each of the 11 benchmark problems is evaluated at multiple levels of difficulty, controlled by the parameter $m$ (the number of objectives). The $m$-values used are:

- **3 objectives**
- **5 objectives**
- **8 objectives**
- **10 objectives**
- **15 objectives**

Increasing $m$ makes the optimization problem more challenging, as the Pareto front becomes higher-dimensional and the search space grows exponentially. This tests the scalability and robustness of each algorithm.

For each problem and each $m$ value, the algorithms must:
- Find a well-distributed set of solutions on the Pareto front
- Satisfy all constraints (for feasible solutions)
- Handle disconnected or complex feasible regions (especially in DC-type problems)

By comparing performance across these $m$ values, the study reveals how each algorithm copes with increasing problem complexity and dimensionality—a key aspect in real-world multi-objective optimization.


## What is inside this repo

At a high level the repository contains:

1. Raw algorithm result files in JSON format.
2. Aggregated result bundles for each algorithm family.
3. A small reporting script (`test.py`) that can reshape the JSON into Markdown tables.
4. A static HTML report viewer (`main.html`) with pre-embedded tables.
5. LaTeX, plots, logs, and screenshots used while preparing the paper.

This means the repo is best understood as a benchmark analysis archive rather than an installable software package.

## Setup

Use a virtual environment so the analysis dependencies do not leak into the system Python.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

source .venv/bin/activate
```

Once dependencies are installed, you can inspect the static viewer with a browser or run the reporting script if you have the expected input layout in place.

## How the repository is organized

### Top-level files

- [README.md](README.md): This documentation.
- [requirements.txt](requirements.txt): Python packages used for analysis, plotting, JSON handling, and report generation.
- [test.py](test.py): A lightweight report generator that aggregates per-problem JSON into Markdown tables.
- [main.html](main.html): A browser-based static dashboard that embeds precomputed tables for quick viewing.

### Results folders

The numbered result folders are the most important parts of the repo.
- `1-Results-IDBEA/`: JSON outputs for I-DBEA runs.
- `2-Results-NSGA/`: JSON outputs for C-NSGA-III runs and aggregate NSGA result files.
- `3-Results-CMOEA-DD/`: JSON outputs for C-MOEA/DD runs.
- `4-Results-CMOEA/`: JSON outputs for CMOEA runs.
- `5-Results-CMOEA-D/`: JSON outputs for C-MOEA/D runs.
- `6-Results-CTAEA/`: JSON outputs for C-TAEA runs.
- `7-Results-Ctaea-Org/`: Additional CTAEA-related outputs or source results.

Each of these folders stores files named by problem and dimension, for example `C1DTLZ1_m3_I-DBEA.json` or `DC2DTLZ3_m15_C-NSGA-III.json`. The naming pattern makes it easy to trace one algorithm across multiple benchmark problems and different $m$ values.

Organized as `{Problem Name}_{m-value}_{Model Name}.json`


### Analysis and paper-support folders

- `Latex/`: LaTeX source, build artifacts, and exported figures for the paper draft.
- `Papers/`: The research references and paper PDFs used while preparing the analysis.
- `Logs/`: Build or run logs; useful for tracing generated plots or report builds.
- `Final-Accumulation/`: Final consolidated outputs from the analysis pipeline.
- `CTAEA-4/` and `CTAEA-Curr/`: Intermediate and Current working snippets for this whole paper implementation.

### Visual artifacts

The `Latex/` folder contains many generated figures and paper-build files, which tells us this repo was used to prepare a results-heavy paper. The images there are not random clutter; they are the visual outputs that correspond to the final tables and comparisons:

- `best_worst_runs.png`: highlights extremes in the runs.
- `c_vs_dc_problems.png`: compares the C-class and DC-class benchmark families.
- `convergence_profile.png`: convergence behavior across iterations or generations.
- `feasibility_rate.png`: feasibility performance across algorithms and problems.
- `heatmap_metrics.png`: compact metric comparison across the benchmark suite.
- `igd_hv_comparison.png`: side-by-side IGD and HV comparison.
- `summary_table.png`: rendered summary table image for paper inclusion.
- `time_vs_quality.png`: runtime-versus-quality comparison.

The presence of `main.pdf`, `main.tex`, and LaTeX auxiliary files indicates the paper was compiled inside the repo, not just written elsewhere.

## What the data looks like

The raw experiment JSONs follow a consistent nested structure. A sample file contains:

- problem name at the top level, such as `C1DTLZ1`.
- `m` value as the next level, such as `3`, `5`, `8`, `10`, or `15`.
- algorithm name at the third level, such as `I-DBEA` or `C-NSGA-III`.
- metric fields at the leaf level.

Typical leaf fields are:

- `igd_median`
- `igd_iqr`
- `hv_median`
- `hv_iqr`
- `n_runs_feasible`
- `n_runs_total`

This structure is good for analysis because it lets one file represent the full result set for one algorithm or one experiment slice while keeping problem/dimension comparisons simple.

### Meaning of the fields

- `igd_median`: median Inverted Generational Distance across repeated runs. Lower is better.
- `igd_iqr`: spread of IGD across runs. Lower is more stable.
- `hv_median`: median Hypervolume. Higher is better.
- `hv_iqr`: spread of HV across runs.
- `n_runs_feasible`: count of runs that ended in feasible solutions.
- `n_runs_total`: total number of runs, typically `51`.

An important detail from the actual data is that some entries use `Infinity` or `0.0` to signal infeasible or undefined outcomes. That means any table or plot needs to treat those values carefully instead of ranking them as ordinary numeric results.


## Metrics Used

- **IGD (Inverted Generational Distance):** Measures how close the solutions found by an algorithm are to the true Pareto front. Lower IGD means better convergence and diversity.
- **HV (Hypervolume):** Calculates the volume covered by the obtained solutions in the objective space. Higher HV means better spread and quality of solutions.

These metrics are computed using the scripts in the `Metrics/` folder (see `igd.py` and `hv.py` in `CTAEA-Curr/Metrics/` and `CTAEA-4/Metrics/`).


## Novelty

- **Parallel execution:** The code supports running many experiments at once using multiple CPU cores. This makes large-scale testing much faster and is easy to use with the `run_sequential.py --workers N` option.
- **Monte Carlo HV estimation:** For problems with many objectives or constraints, the code uses Monte Carlo simulation to estimate the hypervolume efficiently, making it practical for high-dimensional cases.
- **Unified framework:** All six algorithms (CTAEA and peers) are implemented in a single codebase, making it easy to compare, extend, or benchmark them under the same settings.



## Running the Code

1. **Clone the repository**
	- `git clone https://github.com/rajm012/Quant-Finance-Project`
	- Choose either `CTAEA-Curr` (current, more optimized) or `CTAEA-4` (previous stable version) for your experiments.

2. **Set up the environment**
	- Follow the setup instructions above to create and activate a Python virtual environment, then install dependencies with `pip install -r requirements.txt`.

3. **Navigate to the code folder**
	- `cd CTAEA-Curr` (or `cd CTAEA-4`)

4. **Main scripts and their usage:**

	- **main.py**: Entry point for running a full experiment or a quick test. Edit this file to set up your experiment configuration (algorithm, problem, m-value, etc.). Running this script will execute the selected algorithm(s) and output results to the appropriate results folder.
	  - *Usage:* `python main.py`
	  - *Output:* JSON files with results for each algorithm/problem/m combination.

	- **run_sequential.py**: Runs experiments sequentially or in parallel using multiple CPU cores. You can specify the number of workers for parallel execution.
	  - *Usage:* `python run_sequential.py --workers 4`
	  - *Output:* Aggregated results for all specified runs, saved as JSON files. Great for large-scale benchmarking.
      *Recommended*

	- **run_parallel_gpu.py**: Similar to `run_sequential.py` but optimized for running experiments on multiple GPUs (if available). Use this if you have GPU resources and want to speed up computation.
	  - *Usage:* `python run_parallel_gpu.py --workers 2`
	  - *Output:* Parallelized results leveraging GPU acceleration.

	- **quick_test.py**: Runs a very fast, small-scale test to check if the code and environment are set up correctly. Useful for debugging or verifying installation.
	  - *Usage:* `python quick_test.py`
	  - *Output:* Console output and a small result file, confirming the code runs as expected.

	- **test_parallel.py**: Script for testing the parallel execution logic. Use this to verify that parallelization is working on your system.
	  - *Usage:* `python test_parallel.py`
	  - *Output:* Console output showing parallel task execution.

5. **Code structure**
	- Algorithms are implemented in `Algorithms/` (see `ctaea.py` for CTAEA and `peer_algorithms.py` for others).
	- Problems are defined in `Problems/` (each problem as a separate file).
	- Metrics (IGD, HV, etc.) are in `Metrics/`.
	- Experiment runners and analysis scripts are in `Experiments/` and `Analysis/`.

6. **Results and outputs**
	- After running experiments, results are saved as JSON files in the results folders. You can analyze these using the provided scripts or generate summary tables for reporting.

**Tip:** Edit the configuration or parameters in the main scripts to select which algorithms, problems, and m-values to run. The modular structure makes it easy to add new problems or algorithms for further research.

**Recommendation:** Use the file `run_sequential.py` as it has been most optimized for small systems and can be extended to large too.


### Command-line Parameters (Paras used)

The following command-line arguments are available for the main experiment runner (`run_sequential.py`). You can combine them to control which algorithms, problems, and settings are used:

- `--all` : Run all 6 models sequentially (recommended for full benchmarking).
- `--algo <algorithm>` : Run a specific algorithm only. Choices: `C-TAEA`, `C-NSGA-III`, `I-DBEA`, `C-MOEA/D`, `C-MOEA/DD`, `CMOEA`.
- `--models <indices>` : Run specific models by index (0-5), e.g., `--models 0,1,2` or `--models 0-2`.
- `--workers <n>` : Number of parallel workers (CPU cores) to use within each model. Default: 16. Example: `--workers 8`.
- `--runs <n>` : Number of independent runs per configuration (problem/m/algorithm). Default: 51. Example: `--runs 3` for quick tests.
- `--problems <list>` : List of problem names to run. Default: all 11 problems. Example: `--problems C1DTLZ1 C1DTLZ3`.
- `--m-values <list>` : List of m-values (number of objectives) to use. Default: 3 5 8 10 15. Example: `--m-values 3 5`.
- `--output <dir>` : Output directory for results. Default: `Results_Sequential`.
- `--no-resume` : Do not resume; overwrite existing results (by default, the script resumes incomplete runs).
- `--trace-metrics` : Record per-stage IGD/HV metrics for C-TAEA (slower, for detailed analysis).
- `--list` : List the model execution order and exit (no experiments are run).
- `--status` : Check the status of existing runs in the output directory and exit.

**Examples:**

- Run all models with 16 workers: `python run_sequential.py --all --workers 16`
- Run only C-TAEA: `python run_sequential.py --algo C-TAEA --workers 8`
- Run the first two models: `python run_sequential.py --models 0,1 --workers 4`
- Quick test with 3 runs: `python run_sequential.py --all --runs 3 --workers 4`
- Run only selected problems and m-values: `python run_sequential.py --all --problems C1DTLZ1 C1DTLZ3 --m-values 3 5`

See the script's help (`python run_sequential.py --help`) for full details and more usage examples.

*For proper optimized run, use it like `python run_sequential.py --algo ALGONAME --runs 51 --workers 12 --output DIRNAME`*


## References

- Wang, Y., Li, B., & Yao, X. (2020). A Two-Archive Evolutionary Algorithm for Constrained Multi-Objective Optimization. *IEEE Transactions on Evolutionary Computation*, 24(2), 189-203. [PDF](Papers/Main-Paper.pdf) | [IEEE Xplore](https://ieeexplore.ieee.org/document/9099352)
- Deb, K., Thiele, L., Laumanns, M., & Zitzler, E. (2005). Scalable Test Problems for Evolutionary Multiobjective Optimization. In *Evolutionary Multiobjective Optimization* (pp. 105-145). [DTLZ paper](https://ieeexplore.ieee.org/document/1290934)
- Tian, Y., Zhang, X., Zhang, C., & Jin, Y. (2016). A Multiobjective Evolutionary Algorithm Based on Dominance and Decomposition for Multiobjective Optimization Problems with Complicated Pareto Sets. *IEEE Transactions on Evolutionary Computation*, 20(3), 405-420. [DC-DTLZ paper](https://ieeexplore.ieee.org/document/7553457)
- Other algorithm papers: See links in the "Models or Algorithms" section above for IDBEA, NSGA-III, CMOEA, CMOEA/D, and CMOEA/DD.


