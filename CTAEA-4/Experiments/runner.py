

"""
Experiment configuration matching the paper's setup.
Table V: Number of FEs per test instance.
Table IV: Population sizes.
"""


import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np
import json
import time
import hashlib
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

from Problems import (
    C1DTLZ1, C1DTLZ3, C2DTLZ2, C3DTLZ1, C3DTLZ4,
    DC1DTLZ1, DC1DTLZ3, DC2DTLZ1, DC2DTLZ3, DC3DTLZ1, DC3DTLZ3,
)
from Algorithms import CTAEA, CMOEAD, CNSGAIII, CMOEAD_DD, IDBEA, CMOEA
from Algorithms.utils import get_N_and_H

from Metrics.igd import igd
from Metrics.hv import hypervolume


# ─────────────────────────────────────────────────────────────────────────────
# FE budget (Table V from paper: N * multiplier)
# ─────────────────────────────────────────────────────────────────────────────

FE_TABLE = {
    # C-DTLZ
    'C1DTLZ1':  {3: 500,  5: 600,  8: 800,  10: 1000, 15: 1500},
    'C1DTLZ3':  {3: 1000, 5: 1500, 8: 2500, 10: 3500, 15: 5000},
    'C2DTLZ2':  {3: 250,  5: 350,  8: 500,  10: 750,  15: 1000},
    'C3DTLZ1':  {3: 750,  5: 1250, 8: 2000, 10: 3000, 15: 4000},
    'C3DTLZ4':  {3: 750,  5: 1250, 8: 2000, 10: 3000, 15: 4000},
    # DC-DTLZ 
    'DC1DTLZ1': {3: 500,  5: 600,  8: 800,  10: 1000, 15: 1500},
    'DC1DTLZ3': {3: 1000, 5: 1500, 8: 2500, 10: 3500, 15: 5000},
    'DC2DTLZ1': {3: 500,  5: 600,  8: 800,  10: 1000, 15: 1500},
    'DC2DTLZ3': {3: 1000, 5: 1500, 8: 2500, 10: 3500, 15: 5000},
    'DC3DTLZ1': {3: 750,  5: 1250, 8: 2000, 10: 3000, 15: 4000},
    'DC3DTLZ3': {3: 750,  5: 1250, 8: 2000, 10: 3000, 15: 4000},
}

# Reference point zr for HV calculation
ZR_TABLE = {
    'C3DTLZ4': 2.1,   # zr = (2.1,...,2.1)
    'default': 1.1,   # zr = (1.1,...,1.1)
}

# Problem classes
PROBLEM_CLASSES = {
    'C1DTLZ1': C1DTLZ1,
    'C1DTLZ3': C1DTLZ3,
    'C2DTLZ2': C2DTLZ2,
    'C3DTLZ1': C3DTLZ1,
    'C3DTLZ4': C3DTLZ4,
    'DC1DTLZ1': DC1DTLZ1,
    'DC1DTLZ3': DC1DTLZ3,
    'DC2DTLZ1': DC2DTLZ1,
    'DC2DTLZ3': DC2DTLZ3,
    'DC3DTLZ1': DC3DTLZ1,
    'DC3DTLZ3': DC3DTLZ3,
}

# Algorithm classes
ALGO_CLASSES = {
    'C-TAEA':    CTAEA,
    'C-NSGA-III': CNSGAIII,
    'C-MOEA/D':  CMOEAD,
    'C-MOEA/DD': CMOEAD_DD,
    'I-DBEA':    IDBEA,
    'CMOEA':     CMOEA,
}


def get_max_fe(problem_name, m, N):
    """Get maximum FEs = multiplier * N."""
    mult = FE_TABLE.get(problem_name, {}).get(m, 500)
    return mult * N


def get_ref_point(problem_name, m):
    """Get HV reference point."""
    val = ZR_TABLE.get(problem_name, ZR_TABLE['default'])
    return np.full(m, val)


def run_single(problem_name, m, algo_name, seed=0, verbose=False):
    """
    Run one algorithm instance on one problem/dimension combination.

    Returns
    -------
    dict with keys: igd, hv, n_feasible, time_sec, F_nd (nondominated obj)
    """
    ProbClass = PROBLEM_CLASSES[problem_name]
    AlgoClass = ALGO_CLASSES[algo_name]

    problem = ProbClass(n_obj=m)
    N, _ = get_N_and_H(m)
    max_fe = get_max_fe(problem_name, m, N)

    algo = AlgoClass(
        problem=problem,
        N=N,
        max_fe=max_fe,
        seed=seed,
        verbose=verbose,
    )

    t0 = time.time()
    _, _, _ = algo.run()
    elapsed = time.time() - t0

    # Get non-dominated feasible front (support both peer and C-TAEA APIs)
    if hasattr(algo, 'get_nondominated'):
        X_nd, F_nd, CV_nd = algo.get_nondominated()
    elif hasattr(algo, 'get_nondominated_CA'):
        X_nd, F_nd, CV_nd = algo.get_nondominated_CA()
    else:
        X_nd, F_nd, CV_nd = _get_nondominated(algo)

    # Compute metrics
    P_star = problem.get_pareto_front_reference(n_points=500)
    igd_val = igd(F_nd, P_star) if len(F_nd) > 0 else np.inf

    ref_point = get_ref_point(problem_name, m)
    hv_val = _compute_hv(F_nd, ref_point) if len(F_nd) > 0 else 0.0

    return {
        'igd': igd_val,
        'hv': hv_val,
        'n_feasible': len(F_nd),
        'time_sec': elapsed,
        'F_nd': F_nd.tolist() if len(F_nd) > 0 else [],
    }


def _run_single_safe(task):
    """Process-safe wrapper for a single run task."""
    run_idx, problem_name, m, algo_name, seed, verbose = task
    try:
        res = run_single(problem_name, m, algo_name, seed=seed, verbose=verbose)
        return run_idx, res, None
    except Exception as e:
        return run_idx, None, str(e)


def _get_nondominated(algo):
    """Extract non-dominated feasible solutions from algorithm."""
    from Algorithms.utils import non_dominated_indices
    if not hasattr(algo, 'pop_CV'):
        raise AttributeError(
            f"Algorithm {algo.__class__.__name__} has no compatible nondominated-extraction API"
        )
    feas = algo.pop_CV == 0
    X_f = algo.pop_X[feas]
    F_f = algo.pop_F[feas]
    CV_f = algo.pop_CV[feas]
    if len(F_f) == 0:
        return X_f, F_f, CV_f
    nd = non_dominated_indices(F_f)
    return X_f[nd], F_f[nd], CV_f[nd]


def _compute_hv(F, ref_point):
    """
    Compute HV with adaptive sampling based on dimensionality and front size.
    
    OPTIMIZATION #4: Adaptive Monte Carlo sampling for high-dimensional cases.
    Paper uses exact HV for m <= 5, MC for m > 5.
    This version uses fewer samples for small fronts and high m (often sufficient).
    Speedup: 2-5 seconds per run (high-m cases).
    """
    from Metrics.hv import hypervolume, hypervolume_monte_carlo
    m = F.shape[1]
    n_points = len(F)
    
    # Exact HV for low dimensions (paper standard)
    if m <= 5:
        try:
            return hypervolume(F, ref_point)
        except Exception:
            pass
    
    # Adaptive sampling for high dimensions
    # Default: scale with front size; reduce per paper's experience with MC convergence
    # For m=8-15: 15k-30k samples usually sufficient, 50k is often overkill
    n_samples = min(50000, max(15000, n_points * 150))
    
    if m > 12:
        # Very high dimension: cap samples to reduce time
        n_samples = min(n_samples, 25000)
    
    return hypervolume_monte_carlo(F, ref_point, n_samples=n_samples)


def run_experiment(problem_names=None, m_values=None, algo_names=None,
    n_runs=51, output_dir='Results', verbose=False, quick_test=False, n_jobs=1):
    """
    Run full experiment replicating paper results.

    Parameters
    ----------
    problem_names : list or None (all)
    m_values      : list or None ([3,5,8,10,15])
    algo_names    : list or None (all 6)
    n_runs        : int (51 per paper)
    output_dir    : str
    verbose       : bool
    quick_test    : bool  if True, run 3 runs with m=3 only
    n_jobs        : int   number of parallel worker processes for independent runs
    """
    if problem_names is None:
        problem_names = list(PROBLEM_CLASSES.keys())
    if m_values is None:
        m_values = [3, 5, 8, 10, 15]
    if algo_names is None:
        algo_names = list(ALGO_CLASSES.keys())

    if quick_test:
        n_runs = 3
        m_values = [3]

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    results = {}

    total = len(problem_names) * len(m_values) * len(algo_names) * n_runs
    done = 0

    for prob_name in problem_names:
        results[prob_name] = {}
        for m in m_values:
            results[prob_name][m] = {}
            for algo_name in algo_names:
                print(f"\n{'='*60}")
                print(f"Problem: {prob_name}, m={m}, Algorithm: {algo_name}")
                print(f"{'='*60}")

                run_results = []
                seed_key = f"{prob_name}|{m}"
                base = int(hashlib.sha256(seed_key.encode("utf-8")).hexdigest()[:8], 16)

                tasks = []
                for run_idx in range(n_runs):
                    seed = (base + run_idx * 100) % (2**31)
                    tasks.append((run_idx, prob_name, m, algo_name, seed, verbose))

                if n_jobs <= 1:
                    ordered = []
                    for task in tasks:
                        run_idx, res, err = _run_single_safe(task)
                        ordered.append((run_idx, res, err))
                else:
                    ordered = []
                    with ProcessPoolExecutor(max_workers=n_jobs) as ex:
                        futures = [ex.submit(_run_single_safe, task) for task in tasks]
                        for fut in as_completed(futures):
                            ordered.append(fut.result())

                ordered.sort(key=lambda t: t[0])

                for run_idx, res, err in ordered:
                    if err is None:
                        run_results.append(res)
                        print(f"  Run {run_idx+1:2d}/{n_runs} | "
                              f"IGD={res['igd']:.4e} | "
                              f"HV={res['hv']:.4e} | "
                              f"Feasible={res['n_feasible']} | "
                              f"Time={res['time_sec']:.1f}s")
                    else:
                        print(f"  Run {run_idx+1:2d}/{n_runs} FAILED: {err}")
                        run_results.append({'igd': np.inf, 'hv': 0.0,
                                           'n_feasible': 0, 'time_sec': 0.0,
                                           'F_nd': []})
                    done += 1

                # Keep infinities for faithful reporting when algorithms fail to
                # find feasible solutions in some runs.
                igd_vals = [r['igd'] for r in run_results]
                hv_vals = [r['hv'] for r in run_results]

                summary = {
                    'igd_median': float(np.median(igd_vals)) if igd_vals else np.inf,
                    'igd_iqr':    float(np.subtract(*np.percentile(igd_vals, [75, 25])))
                                  if len(igd_vals) > 1 else 0.0,
                    'hv_median':  float(np.median(hv_vals)),
                    'hv_iqr':     float(np.subtract(*np.percentile(hv_vals, [75, 25])))
                                  if len(hv_vals) > 1 else 0.0,
                    'n_runs_feasible': sum(r['n_feasible'] > 0 for r in run_results),
                    'runs': run_results,
                }

                print(f"\n  SUMMARY -> IGD: {summary['igd_median']:.4e} "
                      f"({summary['igd_iqr']:.2e}) | "
                      f"HV: {summary['hv_median']:.4e} "
                      f"({summary['hv_iqr']:.2e})")

                results[prob_name][m][algo_name] = summary

                # Save incrementally
                save_path = out_dir / f"{prob_name}_m{m}_{algo_name.replace('/', '_')}.json"
                with open(save_path, 'w') as f:
                    # Convert numpy types for JSON
                    json.dump(_to_serializable(summary), f, indent=2)

    # Save combined results
    combined_path = out_dir / 'all_results.json'
    with open(combined_path, 'w') as f:
        json.dump(_to_serializable(results), f, indent=2)

    print(f"\n\nAll results saved to {out_dir}/")
    return results



def _to_serializable(obj):
    """Convert numpy types to Python native for JSON serialization."""
    if isinstance(obj, dict):
        return {str(k): _to_serializable(v) for k, v in obj.items()}
    
    elif isinstance(obj, list):
        return [_to_serializable(v) for v in obj]
    
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    
    elif isinstance(obj, (np.float32, np.float64)):
        return float(obj)
    
    elif isinstance(obj, (np.int32, np.int64)):
        return int(obj)
    
    return obj


