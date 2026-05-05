# CTAEA-Curr vs CTAEA-4: Implementation Comparison (C-TAEA Focus)

Date: 2026-05-02

## Scope
This report compares:
- `CTAEA-Curr`
- `CTAEA-4`

with emphasis on C-TAEA core behavior, speed, and alignment with Li et al. (C-TAEA paper Algorithms 2-5, Table IV/V style replication setup).

---

## Executive Verdict

## Overall
- **For paper-faithful C-TAEA algorithmic behavior:** **CTAEA-Curr is superior**.
- **For strict replication-style HV policy (fixed 50k MC in high dimensions):** **CTAEA-Curr is superior**.
- **For speed in this current code snapshot:** **CTAEA-Curr is massively faster in practice** (see benchmark below).
- **For quick ad-hoc experimentation features:** **CTAEA-Curr is richer** (stage metrics trace support).

## Why this is the verdict
Although `CTAEA-4` includes comments about optimizations, it also includes a **non-paper simplification** of DA update and reintroduces slower loop-based building blocks in other places. In measured runtime, `CTAEA-Curr` outperformed by a very large margin on the same task.

---

## Controlled Side-by-Side Benchmark (same machine, same seed)

Task used:
- Problem: `C1DTLZ3`
- Objectives: `m=3`
- Algorithm: `C-TAEA`
- Seed: `42`
- API: `Experiments.runner.run_single`

Observed output:
- `CTAEA-Curr`: `time=5.927s`, `IGD=8.004956145544167`, `HV=0.0`, `n_feasible=91`
- `CTAEA-4`: `time=237.017s`, `IGD=0.06130054083653569`, `HV=0.1603525020797562`, `n_feasible=91`

Interpretation:
- Runtime difference is extreme in favor of `CTAEA-Curr` for this task.
- Metric values differ substantially, indicating behavioral differences beyond pure performance tuning.

---

## Key Implementation Differences

## 1) C-TAEA Reproduction / Offspring Path

### CTAEA-Curr
- Uses cached `rho_c, rho_d` once per generation and then batched offspring generation.
- Uses batched SBX/mutation path (`sbx_crossover_batch`, `polynomial_mutation_batch`) with option for scalar RNG-matching mode.
- References:
  - `CTAEA-Curr/Algorithms/ctaea.py` (`_generate_offspring`)
  - `CTAEA-Curr/Algorithms/utils.py` (batched operator functions)

### CTAEA-4
- Uses cached rho once/generation as well.
- Uses scalar pair-by-pair SBX/mutation loop.
- References:
  - `CTAEA-4/Algorithms/ctaea.py` (`_generate_offspring`)

### Verdict
- **Speed/throughput:** `CTAEA-Curr` better (batched path).
- **Algorithmic intent:** both are valid in principle for Algorithm 4/5.

---

## 2) DA Update (Algorithm 3) Fidelity

### CTAEA-Curr
- Keeps iterative filling style with subregion capacity logic (`itr`, `slots_needed`, per-subregion non-dominated subset selection).
- This structure is close to paper Algorithm 3 flow.
- Reference:
  - `CTAEA-Curr/Algorithms/ctaea.py` (`_update_DA`)

### CTAEA-4
- Explicitly replaces DA mechanism with a simplified version:
  - single ND sort + partial-front heuristic selection with per-subregion cap.
- File itself labels this as simplification and speed optimization.
- Reference:
  - `CTAEA-4/Algorithms/ctaea.py` (`_update_DA` docstring says simplified version)

### Verdict
- **Paper fidelity:** `CTAEA-Curr` better.
- **Deviation risk:** `CTAEA-4` has a clear algorithmic deviation from original Algorithm 3.

---

## 3) Dominance/Sorting Primitives

### CTAEA-Curr
- `fast_non_dominated_sort`: vectorized dominance matrix.
- `non_dominated_indices`: vectorized.
- Also provides constrained-dominance matrix and vectorized constrained non-dominated sort utility used by peer algorithms.
- Reference:
  - `CTAEA-Curr/Algorithms/utils.py`

### CTAEA-4
- `non_dominated_indices`: vectorized.
- `fast_non_dominated_sort`: loop-based nested implementation.
- Constrained-dominance vectorized helpers removed from exports and peer paths.
- Reference:
  - `CTAEA-4/Algorithms/utils.py`

### Verdict
- **Low-level efficiency and consistency:** `CTAEA-Curr` better.

---

## 4) Metrics Policy (HV) in Experiment Runners

### CTAEA-Curr
- For `m>5`: fixed `n_samples=50000` MC-HV fallback.
- This is consistent with strict replication style if you want stable, paper-like post-processing cost assumptions.
- References:
  - `CTAEA-Curr/Experiments/runner.py` (`_compute_hv`)
  - `CTAEA-Curr/Experiments/parallel_runner.py` (`_compute_hv`)

### CTAEA-4
- Uses adaptive sample count (15k to 50k, and cap 25k for very high m).
- Faster but no longer fixed-sample policy.
- References:
  - `CTAEA-4/Experiments/runner.py` (`_compute_hv`)
  - `CTAEA-4/Experiments/parallel_runner.py` (`_compute_hv`)

### Verdict
- **Replication fidelity:** `CTAEA-Curr` better.
- **Metric speed:** `CTAEA-4` potentially faster in HV step only.

---

## 5) Optional Stage-wise Metric Tracing (10%-100% FE checkpoints)

### CTAEA-Curr
- Supports stage metric tracing directly in CTAEA and runners (`metrics_trace`, `collect_metrics_stages`, `hv_ref_point`).
- Useful for convergence studies and per-stage plots.
- References:
  - `CTAEA-Curr/Algorithms/ctaea.py`
  - `CTAEA-Curr/Experiments/runner.py`
  - `CTAEA-Curr/Experiments/parallel_runner.py`

### CTAEA-4
- Removed this feature from runner interfaces and task/result schema.

### Verdict
- **Research analysis capability:** `CTAEA-Curr` better.

---

## 6) Default `max_fe` Handling Robustness

### CTAEA-Curr
- If `max_fe` is not provided, defaults to `1000*N` in CTAEA class.
- References:
  - `CTAEA-Curr/Algorithms/ctaea.py`
  - `CTAEA-Curr/Algorithms/peer_algorithms.py`

### CTAEA-4
- Sets `self.max_fe = max_fe` directly (no fallback in class).
- If called without runner-provided value, can fail in loop condition (`self.fe_count < self.max_fe`).
- References:
  - `CTAEA-4/Algorithms/ctaea.py`
  - `CTAEA-4/Algorithms/peer_algorithms.py`

### Verdict
- **Robustness/API safety:** `CTAEA-Curr` better.

---

## 7) Code Hygiene Observations

### CTAEA-4
- Duplicate method definition appears in C-TAEA class:
  - `_non_dominated_subset` appears twice.
- Reference:
  - `CTAEA-4/Algorithms/ctaea.py` (around lines 590 and 599)

Impact:
- Functional behavior is not broken (later definition overrides), but it is a maintenance smell.

---

## Paper Deviation Assessment

## CTAEA-Curr
- **Core C-TAEA algorithm structure:** largely aligned with paper Algorithms 2-5.
- **Potential non-paper additions:** stage-wise metric tracing (optional), batched operators (implementation acceleration only, not conceptual change).
- **Replication metrics:** fixed MC samples in high dimensions preserved.

Assessment: **No major conceptual deviation in C-TAEA core.**

## CTAEA-4
- **Major deviation:** DA update changed from iterative Algorithm 3 behavior to simplified heuristic selection.
- **Replication-policy deviation:** adaptive MC-HV sample size replaces fixed 50k fallback.
- **Minor quality concern:** duplicate method definition and missing `max_fe` fallback in class layer.

Assessment: **Yes, there are meaningful deviations from strict paper-faithful behavior.**

---

## Aspect-by-Aspect Superiority Matrix

| Aspect | Superior Folder | Reason |
|---|---|---|
| C-TAEA paper fidelity (Algorithms 2-5) | CTAEA-Curr | Keeps iterative DA update style closer to paper |
| End-to-end speed (current snapshot) | CTAEA-Curr | Measured much faster on same task |
| Operator implementation throughput | CTAEA-Curr | Batched SBX/mutation support |
| Dominance/sorting utilities | CTAEA-Curr | More vectorized primitives and constrained variants |
| Strict replication HV policy | CTAEA-Curr | Fixed 50k MC fallback preserved |
| Stage-wise convergence tracking | CTAEA-Curr | Built-in `metrics_trace` pipeline |
| Simplicity of CTAEA code path | CTAEA-4 | Simpler DA selection logic, fewer moving parts |
| Potential HV post-processing speed only | CTAEA-4 | Adaptive MC sample count |

---

## Final Recommendation

If your priority is:

1. **Paper-faithful replication and publishable comparability:** use **CTAEA-Curr** as base.
2. **Maximum speed with acceptable algorithmic drift:** use `CTAEA-4` only after explicitly documenting DA/HV deviations.
3. **Best practical path:** keep `CTAEA-Curr` core (especially DA update), and port only safe, non-conceptual optimizations if needed.

In short: **CTAEA-Curr is the stronger implementation overall for your stated replication-oriented objective.**
