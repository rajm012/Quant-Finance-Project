# C-TAEA: Two-Archive Evolutionary Algorithm for Constrained Multiobjective Optimization

Replication of:

> Li, K., Chen, R., Fu, G., & Yao, X. (2019).
> *Two-Archive Evolutionary Algorithm for Constrained Multiobjective Optimization.*
> IEEE Transactions on Evolutionary Computation, 23(2), 303–315.

---

## Project Structure

```
CTAEA/
├── Problems/               # Benchmark problems
│   ├── base.py             # Abstract base class
│   ├── C1DTLZ1.py          # Type-1: infeasible barrier
│   ├── C1DTLZ3.py          # Type-1: infeasible ribbon
│   ├── C2DTLZ2.py          # Type-2: disjoint feasible PF segments
│   ├── C3DTLZ1.py          # Type-3: PF = constraint surface (simplex)
│   ├── C3DTLZ4.py          # Type-3: PF = constraint surface (sphere)
│   ├── DC1DTLZ1/3.py       # DC Type-1: cone-shaped feasible strips
│   ├── DC2DTLZ1/3.py       # DC Type-2: oscillating CV landscape
│   └── DC3DTLZ1/3.py       # DC Type-3: combined
│
├── Algorithms/
│   ├── ctaea.py            # C-TAEA (Algorithms 1–5 from paper)
│   ├── peer_algorithms.py  # C-MOEA/D, C-NSGA-III, C-MOEA/DD, I-DBEA, CMOEA
│   └── utils.py            # SBX, poly mutation, weight vectors, dominance
│
├── Metrics/
│   ├── igd.py              # IGD and IGD+
│   ├── hv.py               # Exact HV (WFG) + Monte Carlo approximation
│   └── reference_points.py # Weight vector generation, PF sampling
│
├── Experiments/
│   └── runner.py           # Full experiment runner (Table V FE budgets)
│
├── Analysis/
│   └── analysis.py         # Wilcoxon tests, table generation
│
├── Results/                # Created at runtime
├── main.py                 # Entry point
└── requirements.txt
```

---

## Usage

### Single run
```bash
# C-TAEA on C1-DTLZ3, m=3, 5 runs
python main.py --problem C1DTLZ3 --m 3 --algo C-TAEA --runs 5

# C-NSGA-III on DC2-DTLZ1, m=5
python main.py --problem DC2DTLZ1 --m 5 --algo C-NSGA-III --runs 3
```

Problems: `C1DTLZ1`, `C1DTLZ3`, `C2DTLZ2`, `C3DTLZ1`, `C3DTLZ4`, `DC1DTLZ1`, `DC1DTLZ3`, `DC2DTLZ1`, `DC2DTLZ3`, `DC3DTLZ1`, `DC3DTLZ3`

Algorithms: `C-TAEA`, `C-NSGA-III`, `C-MOEA/D`, `C-MOEA/DD`, `I-DBEA`, `CMOEA`

### Replication (51 runs * 11 problems * 5 m-values * 6 algorithms)
```bash
python main.py --full --output Results
python main.py --analyze --output Results
```

### Fast parallel replication (single-shot)

Run the full paper sweep with process-level parallelism across all runs/configurations:

```bash
python main.py \
  --full \
  --output Results \
  --n-jobs 56 \
  --parallel-mode global \
  --worker-threads 1
```

GPU-aware task pinning (round-robin task assignment to GPU IDs):

```bash
python main.py \
  --full \
  --output Results \
  --n-jobs 56 \
  --parallel-mode global \
  --gpu-ids 0,1,2,3,4,5,6,7 \
  --worker-threads 1
```

Notes:
- `--parallel-mode global` runs all independent runs in one global process pool.
- `--worker-threads 1` avoids BLAS/OpenMP oversubscription at high process counts.
- `--gpu-ids` pins tasks to GPU IDs via `CUDA_VISIBLE_DEVICES` per worker task.
- Current algorithm code is NumPy-based, so speedup mainly comes from multiprocessing; GPU pinning is included to support GPU-backed extensions.

---

## Algorithm Design (C-TAEA)

| Archive | Role | Feasibility |
|---------|------|-------------|
| **CA** (Convergence-oriented) | Drives toward PF, ensures feasibility | Yes — prioritizes feasible solutions |
| **DA** (Diversity-oriented) | Explores under-exploited areas including infeasible regions | No — ignores constraint violations |

### Key algorithms implemented

| Algorithm | Description |
|-----------|-------------|
| **Alg. 1** | Association: assign each solution to a subregion (weight vector) |
| **Alg. 2** | CA update: non-dominated sort + density-based trimming |
| **Alg. 3** | DA update: iterative subregion filling using CA as reference |
| **Alg. 4** | Restricted mating selection: chooses CA/DA based on ρ_c vs ρ_d |
| **Alg. 5** | Tournament selection: feasibility-driven |

### Why C-TAEA outperforms feasibility-driven methods

```
Problem C1-DTLZ3: Infeasible ribbon blocks convergence to PF
───────────────────────────────────────────────────────────
Feasibility-driven methods (C-NSGA-III, C-MOEA/D):
  → Stop at the outer feasible boundary
  → HV = 0, no solutions reach the true PF

C-TAEA:
  → DA ignores the infeasible ribbon, pushes toward PF
  → CA eventually crosses the infeasible barrier via mating from DA
  → HV > 0, solutions reach true PF
```

---

## Benchmark Problems

### C-DTLZ Suite (objective-space constraints)

| Problem | Type | Challenge |
|---------|------|-----------|
| C1-DTLZ1 | Type-1 | Narrow feasible region above PF |
| C1-DTLZ3 | Type-1 | Infeasible ribbon intersects objective space |
| C2-DTLZ2 | Type-2 | Disjoint feasible PF segments (r=0.1) |
| C3-DTLZ1 | Type-3 | PF = constraint surface (hyperplane) |
| C3-DTLZ4 | Type-3 | PF = constraint surface (sphere, biased) |

### DC-DTLZ Suite (decision-space constraints)

| Problem | Type | Challenge |
|---------|------|-----------|
| DC1-DTLZ1/3 | Type-1 | Cone-shaped feasible strips (a=3, b=0.5) |
| DC2-DTLZ1/3 | Type-2 | Oscillating CV; local CV optima far from PF |
| DC3-DTLZ1/3 | Type-3 | Combined: segmented cones + CV oscillation |

---

## Performance Metrics

- **IGD** (Inverted Generational Distance): `IGD(P, P*) = mean_{z∈P*} dist(z, P)`
  - Lower is better
  - Only feasible solutions contribute

- **HV** (Hypervolume): volume dominated by P, bounded by `zr=(1.1,...,1.1)`
  - Higher is better
  - Exact WFG algorithm for m≤5; Monte Carlo for larger m

---

## Experimental Setup (matching paper)

| Parameter | Value |
|-----------|-------|
| Crossover | SBX, p_c=0.9, η_c=30 |
| Mutation | Polynomial, p_m=1/n, η_m=20 |
| Population | Table IV (91–275 depending on m) |
| FE budget | Table V (500N–5000N depending on problem/m) |
| Independent runs | 51 |
| Statistical test | Wilcoxon rank-sum, α=0.05 |

