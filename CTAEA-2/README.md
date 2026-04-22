# C-TAEA: Two-Archive Evolutionary Algorithm for Constrained Multi-Objective Optimization

Full Python replication of:

> Li, K., Chen, R., Fu, G., & Yao, X. (2019).
> Two-Archive Evolutionary Algorithm for Constrained Multiobjective Optimization.
> *IEEE Transactions on Evolutionary Computation*, 23(2), 303–315.

---

## Project Structure

```
CTAEA/
├── Problems/
│   ├── __init__.py         # Problem registry
│   ├── C_DTLZ.py           # C1-DTLZ1, C1-DTLZ3, C2-DTLZ2, C3-DTLZ1, C3-DTLZ4
│   └── DC_DTLZ.py          # DC1-DTLZ1/3, DC2-DTLZ1/3, DC3-DTLZ1/3 (new in paper)
│
├── Metrics/
│   ├── __init__.py
│   ├── igd.py              # IGD metric (Eq. 14)
│   ├── hv.py               # Hypervolume metric (Eq. 15) — WFG + Monte Carlo
│   └── reference_points.py # P* sampling for IGD, z_r for HV
│
├── Algorithms/
│   ├── __init__.py
│   ├── ctaea.py            # C-TAEA (Algorithms 1–5 from paper)
│   └── peer_algorithms.py  # C-MOEA/D, C-NSGA-III, C-MOEA/DD, I-DBEA, CMOEA
│
├── Experiments/
│   └── run_experiments.py  # 51-run experiment runner
│
├── Results/                # Auto-created, stores .json result files
│
├── Analysis/
│   ├── analysis.py         # Statistical tests, table generation
│   └── visualization.py    # Scatter plots, PCP plots, box plots
│
├── utils.py                # Weight vectors, SBX, PM, non-dom sort, etc.
└── test.py                 # Quick smoke-test suite
```

---

## Installation

```bash
pip install numpy scipy matplotlib
```

---

## Quick Start

### 1. Run smoke tests
```bash
cd CTAEA
python test.py --quick
```

### 2. Quick experiment (3 problems, m=3, 3 runs)
```bash
python Experiments/run_experiments.py --quick
```

### 3. Single algorithm on one problem
```bash
python Experiments/run_experiments.py \
    --algorithm C-TAEA \
    --problem C1-DTLZ3 \
    --m 3 \
    --runs 5
```

### 4. Full paper experiment (51 runs, all problems)
```bash
python Experiments/run_experiments.py --runs 51
```

### 5. Generate analysis tables
```bash
python Analysis/analysis.py
```

### 6. Generate figures
```bash
python Analysis/visualization.py
```

---

## Using C-TAEA Directly

```python
import sys
sys.path.insert(0, '/path/to/CTAEA')

from Problems import ALL_PROBLEMS
from Algorithms.ctaea import CTAEA
from Metrics import igd, hypervolume
from Metrics.reference_points import get_reference_points, get_reference_point_hv

# Setup
m       = 3
problem = ALL_PROBLEMS['C1-DTLZ3'](m=m)

# Run C-TAEA
algo = CTAEA(
    problem=problem,
    m=m,
    N=91,            # population size (Table IV)
    max_fe=91*1000,  # function evaluations (Table V)
    seed=42,
    verbose=True
)

CA, DA = algo.run()

# Get non-dominated feasible solutions
PF = algo.get_pareto_front('CA')
print(f"PF points: {len(PF)}")

# Compute metrics
P_star  = get_reference_points('C1-DTLZ3', m)
z_r     = get_reference_point_hv('C1-DTLZ3', m)

igd_val = igd(PF, P_star)
hv_val  = hypervolume(PF, z_r)

print(f"IGD = {igd_val:.4e}")
print(f"HV  = {hv_val:.4e}")
```

---

## Paper Reproducibility Notes

### Benchmark Problems
| Suite   | Problems | Constraint Acts On |
|---------|----------|--------------------|
| C-DTLZ  | C1-DTLZ1, C1-DTLZ3, C2-DTLZ2, C3-DTLZ1, C3-DTLZ4 | Objective space |
| DC-DTLZ | DC1-DTLZ1/3, DC2-DTLZ1/3, DC3-DTLZ1/3 | Decision space |

### Parameter Settings (Table III)
| Parameter | Value |
|-----------|-------|
| SBX crossover prob | 0.9 |
| Mutation prob | 1/n |
| SBX index η_c | 30 |
| Mutation index η_m | 20 |

### Population Sizes (Table IV)
| m | N |
|---|---|
| 3 | 91 |
| 5 | 210 |
| 8 | 156 |
| 10 | 275 |
| 15 | 135 |

### Key Algorithms (from paper)
- **Algorithm 1**: Association (perpendicular distance to weight vectors)
- **Algorithm 2**: CA Update (feasibility + nondominated sorting + subregion crowding)
- **Algorithm 3**: DA Update (diversity without feasibility, iterative subregion filling)
- **Algorithm 4**: Restricted Mating Selection (adaptive CA/DA selection by ρ_c, ρ_d)
- **Algorithm 5**: Tournament Selection (feasibility-driven binary tournament)

### Performance Metrics
- **IGD** (Eq. 14): Lower is better
- **HV** (Eq. 15): Higher is better
- Statistical test: Wilcoxon rank-sum at 5% significance
  - `†` = C-TAEA significantly better
  - `‡` = peer significantly better

---

## Expected Results (from paper, Table I)

C-TAEA should significantly outperform all peers on:
- **C1-DTLZ3**: IGD ~50× lower than peers (only C-TAEA overcomes infeasible barrier)
- **C2-DTLZ2**: Only C-TAEA finds all disjoint feasible PF segments  
- **DC2-DTLZ1/3**: Only C-TAEA finds feasible solutions consistently
- **DC3-DTLZ1/3**: Only C-TAEA achieves non-zero HV

---

## Troubleshooting

**Slow performance**: Use `--m 3` for initial testing. Many-objective cases (m=10,15) with full budget (5000×N FEs) take significant time. Use PyPy or parallel runs.

**No feasible solutions**: Normal for C1-DTLZ3 and DC2 problems with small budgets. The paper uses the full budget (see Table V).

**HV = 0**: Expected for all peer algorithms on C1-DTLZ3 (Table VI in paper). C-TAEA is the only algorithm that achieves positive HV on this problem.
