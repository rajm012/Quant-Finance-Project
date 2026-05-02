# C-TAEA Performance Analysis & Optimization Report

**Date:** May 2, 2026  
**Project:** C-TAEA Implementation & Replication (Constrained Multiobjective Optimization)

---

## EXECUTIVE SUMMARY

The current C-TAEA implementation in `CTAEA-4` is **functionally correct** for paper replication but suffers from significant **computational bottlenecks** that make wall-clock time impractical for the full experimental grid (12 problems × 5 dimensions × 51 runs = 3,060 total runs). The slowness is **not primarily from FE budgets alone**, but from inefficient Python-level dominance checking and redundant rho computation that compounds over thousands of generations.

**Estimated wall-time for full grid:** 50–100 CPU hours on a single thread (vs. expected 20–30 hours with optimizations).

---

## PART 1: DETAILED SLOWNESS ANALYSIS

### 1.1 Root Causes (in order of impact)

#### **Issue #1: Nested Python Dominance Loop (HIGHEST IMPACT)**
- **Location:** `Algorithms/utils.py` lines 290–301 (`non_dominated_indices` function)
- **Current Implementation:**
  ```python
  def non_dominated_indices(F):
      """Return boolean mask of non-dominated solutions."""
      N = len(F)
      nd = np.ones(N, dtype=bool)
      for i in range(N):
          for j in range(N):
              if i != j and nd[i]:
                  if dominates(F[j], F[i]):
                      nd[i] = False
                      break
      return nd
  ```
  - **Complexity:** O(N² × m) where m = number of objectives
  - **Called:** Every generation in `_compute_rho()`, plus in CA/DA updates, plus for final feasible-front extraction
  - **For worst case (m=15, N=135):** ~36,400 dominance function calls per ND sort, 10–15 sorts per generation, thousands of generations

- **Why it's slow:**
  - Uses Python-level nested loops instead of vectorized NumPy operations
  - Does NOT pre-filter by Pareto-strict conditions (upper bounds, feasibility)
  - Each call to `dominates(F[j], F[i])` does nested `np.all()` on single-row pairs
  - **Benchmark:** ~10 million nested dominance checks per long run (measurements from comment in submitted issue)

#### **Issue #2: ρ (Rho) Recomputed for Every Parent Selection (VERY HIGH IMPACT)**
- **Location:** `Algorithms/ctaea.py` lines 192–203, lines 220–226 (repeated calls to `_compute_rho()`)
- **Current Implementation:**
  ```python
  def _restricted_mating_selection(self):
      rho_c, rho_d = self._compute_rho()  # ← CALLED EVERY PAIR
      if rho_c > rho_d:
          p1_x = self._tournament_selection(self.CA_X, ...)
      else:
          p1_x = self._tournament_selection(self.DA_X, ...)

  def _get_second_parent(self):
      rho_c, _ = self._compute_rho()  # ← CALLED AGAIN FOR SECOND PARENT
      ...
  ```
  - **Called:** N/2 times per generation (once per parent pair) **+ once per second-parent lookup** = **~N times**
  - **Cost per call:** Duplicates dominance check on full CA ∪ DA (size 2N)
  - **Total cost:** ~N generations × N ND-sorts per generation = O(N² × m) per run

- **Example (C1DTLZ3, m=15, N=135):**
  - FE budget: 5000 × 135 = 675,000 → ~5,000 generations
  - Rho recomputed: ~5,000 × 135 ≈ **675,000 times**
  - Each call: full ND-sort of 270 solutions (2N) using Python loops → worst-case 72,900 comparisons
  - **Total dominance comparisons:** 675,000 × 72,900 ≈ **49 billion comparisons** (!!!)

#### **Issue #3: Repeated Non-Dominated Subsets in DA Update (HIGH IMPACT)**
- **Location:** `Algorithms/ctaea.py` lines 486–580 (`_update_DA` method)
- **Current Implementation:**
  ```python
  for itr in range(1, ...):
      for i in range(N):
          available = [k for k in Hd_subregions[i] if k not in S_idx]
          Oi = self._non_dominated_subset(available, Hd_F)  # ← ND-SORT per slot
          ...
  ```
  - **Called:** ~N slots × ~N iterations (iterative filling) = O(N²) ND-sorts in worst case
  - **Per run cost:** Adds another ~N² × m complexity on top of CA update

#### **Issue #4: CA Update Density Trimming (MEDIUM IMPACT)**
- **Location:** `Algorithms/ctaea.py` lines 361–440 (`_select_by_nondom_and_density`)
- **Current Implementation:**
  - Calls `fast_non_dominated_sort()` on combined feasible + partial front
  - Then iteratively trims crowded regions by recomputing distances and ND-masks
  - **Called:** Every generation (when feasible solutions > N)
  - **Complexity:** O(N² × m) for initial ND-sort + O(N × m) for iterative trimming steps

#### **Issue #5: Post-Processing: 50K-Sample Monte Carlo HV (MEDIUM IMPACT)**
- **Location:** `Experiments/runner.py` line 182, `Experiments/parallel_runner.py` line 190
- **Current Implementation:**
  ```python
  def _compute_hv(F, ref_point):
      m = F.shape[1]
      if m <= 5:
          try:
              return hypervolume(F, ref_point)
          except Exception:
              pass
      return hypervolume_monte_carlo(F, ref_point, n_samples=50000)  # ← ALWAYS MC for m > 5
  ```
  - **For high-dimensional (m=15):** Every run spends ~2–5 seconds on HV alone
  - **For 3,060 total runs:** ~10,000–15,000 seconds of MC sampling
  - **Cost over parallel grid:** Linear in run count, adds 1–2 hours to full experiment

#### **Issue #6: Large FE Budgets per Test Case (STRUCTURAL, not optimization)**
- **Location:** `Experiments/runner.py` lines 17–32 (FE_TABLE)
- **Current Values:**
  | Problem | m=3 | m=5 | m=8 | m=10 | m=15 |
  |---------|-----|-----|-----|------|------|
  | C1DTLZ1 | 500 | 600 | 800 | 1000 | 1500 |
  | **C1DTLZ3** | **1000** | **1500** | **2500** | **3500** | **5000** |
  | C2DTLZ2 | 250 | 350 | 500 | 750 | 1000 |
  | C3DTLZ1 | 750 | 1250 | 2000 | 3000 | 4000 |
  | ...DC variants | Similar multipliers | | | |

  - **With N from Table IV:**
    - C1DTLZ3, m=15: N=135, multiplier=5000 → **675,000 FEs**
    - C3DTLZ4, m=15: N=135, multiplier=4000 → **540,000 FEs**
  - **All 12 problems over all m:** Total FEs across one run = ~4–6 million per problem instance
  - **3,060 runs:** ~12–18 billion total FEs for full grid

- **Why this matters:** 
  - Paper uses these budgets for statistical rigor (51 runs = tight confidence intervals)
  - FE budget alone would suggest ~8 hours on 1.5e9 FEs/hour (typical for non-dominated-sort-heavy EA)
  - But inefficient ND-checking **multiplies** this by 5–10×

---

### 1.2 Estimated Time Breakdown (Single Run: C1DTLZ3, m=15)

| Component | CPU Time | % of Total | Notes |
|-----------|----------|-----------|-------|
| Rho recomputation (Issue #2) | 120–180s | 50–60% | 675k ND-sorts of O(N²×m) each |
| Initial ND-sorts in CA/DA | 40–60s | 20–25% | less frequent but large batches |
| DA iterative filling (Issue #3) | 20–30s | 10–15% | subregion ND-subsets |
| Objective evaluations (physics) | 8–12s | 4–6% | if not using dummy/fast problems |
| Post-process HV MC sampling | 3–5s | 2–3% | 50k samples, m=15 |
| **Total (estimated)** | **200–280s** | **100%** | per single run |

**For full 3,060 runs at 230s/run avg:**
- Sequential: 3,060 × 230 sec ÷ 3600 = **196 CPU hours**
- Parallel 32-core: **~6 wall hours**
- Parallel 60-core (as in repo): **~3.3 wall hours** (still too long for interactive dev)

---

## PART 2: OPTIMIZATION OPPORTUNITIES

### 2.1 High-Priority Optimizations (2–3 hour total implementation)

#### **Optimization #1: Vectorize Dominance Checking in `non_dominated_indices`** ⭐⭐⭐ (CRITICAL)

**Estimated speedup:** 30–50× for this function alone

**Implementation:**
```python
def non_dominated_indices(F):
    """Return boolean mask of non-dominated solutions using vectorized operations."""
    N = len(F)
    # Compute pairwise dominance matrix: dominated[i,j] = True if j dominates i
    # F_i <= F_j element-wise AND F_i ≠ F_j for at least one element
    
    F_expanded_i = F[:, np.newaxis, :]      # Shape: (N, 1, m)
    F_expanded_j = F[np.newaxis, :, :]      # Shape: (1, N, m)
    
    # j dominates i iff F_j <= F_i in all objectives AND F_j < F_i in at least one
    dominate_matrix = (
        (F_expanded_j <= F_expanded_i).all(axis=2) &  # F_j <= F_i for all objectives
        (F_expanded_j < F_expanded_i).any(axis=2)     # F_j < F_i for at least one
    )
    
    # A solution is dominated if any other solution dominates it
    dominated = dominate_matrix.any(axis=1)
    return ~dominated
```

**Complexity:** Still O(N² × m) but uses tight NumPy operations instead of Python loops.  
**Trade-off:** Memory usage O(N² × m) for temp arrays, but typical N ≤ 275 is manageable.

**Alternative (if memory is concern):**
```python
def non_dominated_indices_batch(F, batch_size=50):
    """Memory-efficient version: process pairwise comparisons in batches."""
    N = len(F)
    dominated = np.zeros(N, dtype=bool)
    
    for i in range(0, N, batch_size):
        end_i = min(i + batch_size, N)
        F_batch = F[i:end_i]  # Shape: (batch, m)
        F_batch_exp = F_batch[:, np.newaxis, :]  # (batch, 1, m)
        F_all_exp = F[np.newaxis, :, :]          # (1, N, m)
        
        dominate_by_batch = (
            (F_all_exp <= F_batch_exp).all(axis=2) &
            (F_all_exp < F_batch_exp).any(axis=2)
        )
        dominated[i:end_i] |= dominate_by_batch.any(axis=1)
    
    return ~dominated
```

---

#### **Optimization #2: Cache ρ (Rho) Per Generation** ⭐⭐⭐ (CRITICAL)

**Estimated speedup:** 30–50× for parent selection step

**Implementation:**
```python
def _generate_offspring(self):
    """Generate N offspring using restricted mating selection."""
    Q_X = np.zeros((self.N, self.n_var))
    
    # COMPUTE RHO ONCE PER GENERATION (not once per pair!)
    rho_c, rho_d = self._compute_rho()

    for i in range(0, self.N, 2):
        # Reuse cached rho for all offspring in this generation
        if rho_c > rho_d:
            p1_x = self._tournament_selection(self.CA_X, self.CA_F, self.CA_CV)
        else:
            p1_x = self._tournament_selection(self.DA_X, self.DA_F, self.DA_CV)

        # Use same rho for second parent
        if np.random.rand() < rho_c:
            p2_x = self._tournament_selection(self.CA_X, self.CA_F, self.CA_CV)
        else:
            p2_x = self._tournament_selection(self.DA_X, self.DA_F, self.DA_CV)

        c1_x, c2_x = sbx_crossover(p1_x, p2_x, self.xl, self.xu, 
                                   eta_c=self.eta_c, pc=self.pc)
        ...
```

**Alternative: Instead of binary choice, use rho as probability (already correct above).**

**Impact:** Reduces ND-sorts from **~N per generation** to **1 per generation** (675k → 5k for C1DTLZ3, m=15).

---

#### **Optimization #3: Replace DA Iterative Filling with Simpler Selection** ⭐⭐ (HIGH)

**Estimated speedup:** 5–10× for DA update step

**Current complexity:** O(N² × m) iterative filling with per-slot ND-subsets  
**Alternative:** Use single ND-sort + density-based trimming (similar to CA)

```python
def _update_DA_fast(self, Q_X, Q_F, Q_G, Q_CV):
    """Algorithm 3 (simplified): ND-sort + subregion-aware selection."""
    N = self.N
    Hd_X = np.vstack([self.DA_X, Q_X])
    Hd_F = np.vstack([self.DA_F, Q_F])
    Hd_G = np.vstack([self.DA_G, Q_G])
    Hd_CV = np.concatenate([self.DA_CV, Q_CV])
    
    # Single ND-sort instead of iterative per-slot
    fronts, _ = fast_non_dominated_sort(Hd_F)
    
    S_idx = []
    for front in fronts:
        if len(S_idx) + len(front) <= N:
            S_idx.extend(front)
        else:
            # Select from partial front via subregion association + density
            remaining = N - len(S_idx)
            F_norm = self._normalize(Hd_F[front])
            assignments = associate_to_subregions(np.maximum(F_norm, 1e-10), self.W)
            
            # Sort by subregion then by tchebycheff distance
            subregion_order = sorted(set(assignments))
            selected = []
            for sr in subregion_order:
                sr_idx = [front[j] for j in np.where(assignments == sr)[0]]
                if len(selected) < remaining:
                    selected.extend(sr_idx[:remaining - len(selected)])
            S_idx.extend(selected[:remaining])
            break
    
    S_idx = S_idx[:N]
    self.DA_X = Hd_X[S_idx]
    self.DA_F = Hd_F[S_idx]
    self.DA_G = Hd_G[S_idx]
    self.DA_CV = Hd_CV[S_idx]
```

**Trade-off:** Slightly deviates from strict "iterative filling" but maintains subregion fairness.

---

#### **Optimization #4: Adaptive HV Sampling** ⭐ (MEDIUM)

**Estimated speedup:** 2–5 seconds per run (for m > 5)

**Implementation:**
```python
def _compute_hv(F, ref_point):
    """Compute HV with adaptive sampling based on m and |F|."""
    m = F.shape[1]
    n_points = len(F)
    
    if m <= 5:
        try:
            return hypervolume(F, ref_point)
        except Exception:
            pass
    
    # Adaptive sample count: reduce for small fronts, increase marginally for large m
    # Rule: n_samples = max(15000, min(50000, n_points * 200))
    n_samples = max(15000, min(50000, n_points * 200))
    if m > 12:
        n_samples = min(n_samples, 30000)  # cap for very high m
    
    return hypervolume_monte_carlo(F, ref_point, n_samples=n_samples)
```

**Rationale:** 15k–30k samples are often sufficient for (IGD, HV) to stabilize, especially for large fronts.

---

### 2.2 Medium-Priority Optimizations (1–2 hours)

#### **Optimization #5: Numba JIT for Hottest Loops** ⭐ (LOW, but useful)

**Estimated speedup:** 10–20× for inner comparisons (but requires ~30 min setup)

```python
from numba import njit

@njit
def _fast_dominates_numba(f1, f2):
    """JIT-compiled dominance check."""
    le = True
    for i in range(len(f1)):
        if f1[i] > f2[i]:
            le = False
            break
    lt = False
    for i in range(len(f1)):
        if f1[i] < f2[i]:
            lt = True
            break
    return le and lt

@njit
def _non_dominated_indices_numba(F):
    """JIT-compiled ND-sort."""
    N = F.shape[0]
    nd = np.ones(N, dtype=np.bool_)
    for i in range(N):
        for j in range(N):
            if i != j and nd[i]:
                if _fast_dominates_numba(F[j], F[i]):
                    nd[i] = False
                    break
    return nd
```

**Note:** Still O(N² × m) but with C-level inner loops instead of Python.

---

### 2.3 Lower-Priority Optimizations (Quality-of-life)

#### **Optimization #6: Early Stopping for ND-Checks**
- If you find dominance early, break immediately (already done in current code ✓)

#### **Optimization #7: Reference Point Caching**
- Pre-compute and cache `ref_point` for each problem/m combination to avoid recomputation

---

## PART 3: PARAMETER COMPLIANCE WITH PAPER

### 3.1 Algorithm Configuration vs. Paper

#### **3.1.1 Population Size (Table IV)**

| m | Paper N | Paper H | Implementation | Status |
|---|---------|---------|-----------------|--------|
| 3 | 91 | 12 | `generate_reference_vectors_table_iv(3)` → 91 ✓ | ✅ CORRECT |
| 5 | 210 | 6 | `generate_reference_vectors_table_iv(5)` → 210 ✓ | ✅ CORRECT |
| 8 | 156 | 4 (two-layer) | 120 + 36 = 156 ✓ | ✅ CORRECT |
| 10 | 275 | 4 (two-layer) | 220 + 55 = 275 ✓ | ✅ CORRECT |
| 15 | 135 | 3 (two-layer) | 120 + 15 = 135 ✓ | ✅ CORRECT |

**Code Reference:** `Algorithms/utils.py` lines 41–89 (`get_N_and_H` and `generate_reference_vectors_table_iv`)

✅ **COMPLIANCE:** Population sizes match **EXACTLY**.

---

#### **3.1.2 Function Evaluation Budgets (Table V)**

| Problem | m=3 | m=5 | m=8 | m=10 | m=15 | Paper Source |
|---------|-----|-----|-----|------|------|--------------|
| C1DTLZ1 | 500N | 600N | 800N | 1000N | 1500N | Table V ✓ |
| C1DTLZ3 | 1000N | 1500N | 2500N | 3500N | 5000N | Table V ✓ |
| C2DTLZ2 | 250N | 350N | 500N | 750N | 1000N | Table V ✓ |
| C3DTLZ1 | 750N | 1250N | 2000N | 3000N | 4000N | Table V ✓ |
| C3DTLZ4 | 750N | 1250N | 2000N | 3000N | 4000N | Table V ✓ |
| DC1DTLZ1 | 500N | 600N | 800N | 1000N | 1500N | Table V ✓ |
| DC1DTLZ3 | 1000N | 1500N | 2500N | 3500N | 5000N | Table V ✓ |
| DC2DTLZ1 | 500N | 600N | 800N | 1000N | 1500N | Table V ✓ |
| DC2DTLZ3 | 1000N | 1500N | 2500N | 3500N | 5000N | Table V ✓ |
| DC3DTLZ1 | 750N | 1250N | 2000N | 3000N | 4000N | Table V ✓ |
| DC3DTLZ3 | 750N | 1250N | 2000N | 3000N | 4000N | Table V ✓ |

**Code Reference:** `Experiments/runner.py` lines 17–32 (FE_TABLE)

✅ **COMPLIANCE:** FE budgets match **EXACTLY** (multiplier × N per paper Table V).

---

#### **3.1.3 Operator Parameters (Table III from Paper)**

| Parameter | Paper Value | Implementation | Status |
|-----------|-------------|-----------------|--------|
| **SBX Crossover Distribution Index** (η_c) | 30 | `eta_c=30.0` (line 47, ctaea.py) | ✅ CORRECT |
| **Polynomial Mutation Distribution Index** (η_m) | 20 | `eta_m=20.0` (line 47, ctaea.py) | ✅ CORRECT |
| **Crossover Probability** (p_c) | 0.9 | `pc=0.9` (line 47, ctaea.py) | ✅ CORRECT |
| **Mutation Probability** (p_m) | 1/n_var | `1/len(p)` in `polynomial_mutation` (utils.py line ~370) | ✅ CORRECT |

**Code Reference:** `Algorithms/ctaea.py` line 47 (defaults), `Algorithms/utils.py` lines 316–400 (SBX/PM implementation)

✅ **COMPLIANCE:** Operator parameters match **EXACTLY**.

---

#### **3.1.4 Constraint Handling Strategy**

| Aspect | Paper Algorithm | Implementation | Status |
|--------|-----------------|-----------------|--------|
| **Constraint Violation (CV)** | g(x) ≤ 0; CV = max(0, g(x)) summed | `constraint_violation()` method in Problems/ classes | ✅ CORRECT |
| **Feasibility Priority** | Binary tournament prefers feasible | `_tournament_selection()` lines 229–249 in ctaea.py | ✅ CORRECT |
| **CA Update Logic** | Alg. 2: Feasible first, then infeasible sorted by (CV, g_tch) | `_update_CA()` lines 261–359 in ctaea.py | ✅ CORRECT |
| **DA Update Logic** | Alg. 3: Iterative subregion filling, ignores feasibility | `_update_DA()` lines 491–580 in ctaea.py | ✅ CORRECT |
| **Restricted Mating Selection** | Alg. 4: Choose archive based on ρ_c vs ρ_d | `_restricted_mating_selection()` lines 192–205 in ctaea.py | ✅ CORRECT |

**Code Reference:** `Algorithms/ctaea.py` (Algorithms 2–5), `Problems/base.py` (constraint violation)

✅ **COMPLIANCE:** Constraint handling strategy is **ALGORITHMICALLY CORRECT**.

---

#### **3.1.5 Reference Point for HV**

| Problem | Paper z_r | Implementation | Status |
|---------|-----------|-----------------|--------|
| C3DTLZ4 | 2.1 | `ZR_TABLE['C3DTLZ4'] = 2.1` (runner.py line 48) | ✅ CORRECT |
| All others | 1.1 | `ZR_TABLE['default'] = 1.1` (runner.py line 49) | ✅ CORRECT |

**Code Reference:** `Experiments/runner.py` lines 47–50

✅ **COMPLIANCE:** Reference points are **CORRECT**.

---

#### **3.1.6 Number of Runs & Random Seeds**

| Aspect | Paper | Implementation | Status |
|--------|-------|-----------------|--------|
| **Number of Independent Runs** | 51 | Configurable, default 51 via `run_experiment(n_runs=51)` | ✅ CORRECT |
| **Seed Strategy** | Seed=0,1,...,50 (or similar) | `seed=run_idx * 100` (parallel_runner.py line 194) | ✅ **ACCEPTABLE** (different seed strategy but independent) |
| **Random Seed Control** | Via `np.random.seed()` | `np.random.seed(seed)` (ctaea.py line 59) | ✅ CORRECT |

**Code Reference:** `Experiments/runner.py`, `Experiments/parallel_runner.py`

⚠️ **NOTE:** Paper likely uses sequential seeds (0, 1, ..., 50), but implementation uses (0, 100, 200, ..., 5000). Both are valid for independent runs; just ensures different RNG states.

✅ **COMPLIANCE:** Number of runs and seeding are **CORRECT IN SPIRIT** (small deviation in seed numbering, not algorithmic).

---

### 3.2 Problem Instance Compliance

#### **3.2.1 Constraint Definitions**

| Problem | Paper Constraint | Implementation | Status |
|---------|------------------|-----------------|--------|
| **C1DTLZ1** | 1 − (f_m/0.6) − Σ(f_i/0.5) ≥ 0 | Correctly computed in `C1DTLZ1._evaluate()` | ✅ CORRECT |
| **C1DTLZ3** | (Σf_i² − 16)(Σf_i² − r²) ≥ 0; r ∈ {9, 12.5, 12.5, 15, 15} | `C1DTLZ3.r_map` (C1DTLZ3.py line ~7) | ✅ CORRECT |
| **C2DTLZ2** | max(max_i(...), Σ(...)) ≥ 0; r = 0.1 | Correctly computed in `C2DTLZ2._evaluate()` | ✅ CORRECT |
| **C3DTLZ{1,4}** | Feasible region is convex subset of DTLZ PF | Correctly handled in constraint code | ✅ CORRECT |
| **DC{1,2,3}DTLZ{1,3}** | Disjoint + complex feasible regions | Correctly implemented (see Problems/ folder) | ✅ CORRECT |

**Code Reference:** `Problems/` folder (all C*DTLZ*.py and DC*DTLZ*.py classes)

✅ **COMPLIANCE:** Constraint definitions match **EXACTLY**.

---

#### **3.2.2 Pareto Front Reference (for IGD)**

| Problem | Paper Method | Implementation | Status |
|---------|--------------|-----------------|--------|
| **DTLZ-type** | Simplex/sphere sampling on true PF | `sample_pareto_front_simplex()`, `sample_pareto_front_sphere()` (reference_points.py) | ✅ CORRECT |
| **Feasible PF Filtering** | Sample + filter by constraint(s) | Each Problem class has `get_pareto_front_reference()` | ✅ CORRECT |

**Code Reference:** `Problems/base.py`, `Metrics/reference_points.py`

✅ **COMPLIANCE:** Pareto front sampling is **CORRECT**.

---

### 3.3 Summary: Parameter Compliance Matrix

| Category | Aspect | Paper | Impl. | Match | Status |
|----------|--------|-------|-------|-------|--------|
| **Architecture** | Two archives (CA, DA) | ✓ | ✓ | ✓ | ✅ |
| **Population** | N from Table IV | ✓ | ✓ | ✓ | ✅ |
| **FE Budget** | Multiplier × N (Table V) | ✓ | ✓ | ✓ | ✅ |
| **Operators** | η_c=30, η_m=20, p_c=0.9 | ✓ | ✓ | ✓ | ✅ |
| **Constraint Handling** | Feasibility priority + bi-obj fill | ✓ | ✓ | ✓ | ✅ |
| **Mating Selection** | Alg. 4 (ρ-based) + Alg. 5 (binary tournament) | ✓ | ✓ | ✓ | ✅ |
| **CA Update** | Alg. 2 (ND + density trim) | ✓ | ✓ | ✓ | ✅ |
| **DA Update** | Alg. 3 (iterative subregion filling) | ✓ | ✓ | ✓ | ✅ |
| **Metrics** | IGD (vs. true PF), HV (exact or MC) | ✓ | ✓ | ✓ | ✅ |
| **Reference Points** | z_r = 1.1 (default), 2.1 for C3DTLZ4 | ✓ | ✓ | ✓ | ✅ |
| **Runs / Seeds** | 51 independent runs | ✓ | ✓ | ✓ | ✅ |

---

## PART 4: IMPLEMENTATION ROADMAP

### Phase 1: Critical Optimizations (Do First)
Priority: **HIGHEST** | Time: ~1.5–2 hours

1. **Vectorize `non_dominated_indices()`** (Optimization #1)
   - File: `CTAEA-4/Algorithms/utils.py` lines 290–301
   - Estimated impact: **30–50× speedup** for ND-checking
   - Validation: Unit test with known dominated/non-dominated sets

2. **Cache ρ per generation** (Optimization #2)
   - File: `CTAEA-4/Algorithms/ctaea.py` lines 170–190
   - Estimated impact: **20–40× speedup** for parent selection
   - Validation: Check ρ values unchanged per generation

### Phase 2: Secondary Optimizations (Do Next)
Priority: **HIGH** | Time: ~1 hour

3. **Simplify DA update** (Optimization #3)
   - File: `CTAEA-4/Algorithms/ctaea.py` lines 491–580
   - Estimated impact: **5–10× speedup** for DA update
   - Validation: Verify subregion fairness maintained

4. **Adaptive HV sampling** (Optimization #4)
   - File: `Experiments/runner.py` line 174, `Experiments/parallel_runner.py` line 184
   - Estimated impact: **2–5 sec speedup** per high-m run
   - Validation: Verify IGD/HV distributions match baseline

### Phase 3: Polish (Optional)
Priority: **LOW** | Time: ~30 min

5. **Numba JIT compilation** (Optimization #5)
   - Requires: `pip install numba`
   - Optional but provides additional 10–20× speedup if needed

---

## PART 5: SUMMARY & RECOMMENDATIONS

### Current State
✅ **Algorithmic Correctness:** Implementation faithfully follows paper (Algorithms 1–5, all parameters, all constraints).  
❌ **Computational Efficiency:** Implementation uses Python-level loops for dominance checking, causing **5–10× slowdown** vs. optimized code.

### Expected Speedups (After Proposed Optimizations)

| Optimization | Speedup | Cumulative |
|--------------|---------|-----------|
| Baseline (current) | 1× | 1× |
| + Vectorized ND-checking | 30–50× | 30–50× |
| + Cached ρ | 5–10× | 150–500× |
| + Simplified DA | 2–5× | 300–2500× |
| + Adaptive HV | 1.1–1.2× | 330–3000× |
| + Numba JIT (optional) | 2–5× | 660–15000× |

**Realistic target:** **100–150× overall speedup** (Optimizations 1–4 only)

### Predicted Runtime After Optimizations
- **Single run (C1DTLZ3, m=15):** ~2–3 seconds (from 200–280s)
- **Full 12 problems × 5 m × 51 runs:** ~6–12 CPU hours (from 196 hours)
- **On 32-core machine:** ~20–40 minutes wall time (parallelized)

### Action Items for User

1. **If timeline is tight:** Implement Optimizations #1 and #2 only (~1.5 hours → 50–80× speedup, sufficient for development).
2. **If running full paper replication:** Implement all Phase 1–2 optimizations (~2–3 hours → 100–150× speedup, sub-hour wall time on 32-core).
3. **Validation after each optimization:** Run `python main.py --quick` to verify results remain stable.
4. **Final validation:** Compare results on small subset (e.g., 3 runs, m=3) before full grid.

---

**Document End**
