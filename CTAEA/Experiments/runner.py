

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
import multiprocessing as mp
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
    run_idx, problem_name, m, algo_name, seed, verbose, gpu_id = task

    # Optional per-process GPU pinning. This is a no-op for pure NumPy runs,
    # but enables GPU-aware execution if a downstream GPU backend is introduced.
    if gpu_id is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)

    try:
        res = run_single(problem_name, m, algo_name, seed=seed, verbose=verbose)
        return run_idx, res, None
    except Exception as e:
        return run_idx, None, str(e)


def _parse_gpu_ids(gpu_ids):
    """Normalize gpu_ids into a list of ints."""
    if gpu_ids is None:
        return []
    if isinstance(gpu_ids, str):
        cleaned = gpu_ids.strip()
        if not cleaned:
            return []
        return [int(x.strip()) for x in cleaned.split(',') if x.strip()]
    return [int(x) for x in gpu_ids]


def _stable_seed(problem_name, m, algo_name, run_idx):
    """Create stable deterministic seed independent of Python hash randomization."""
    seed_key = f"{problem_name}|{m}|{algo_name}|{run_idx}"
    return int(hashlib.sha256(seed_key.encode('utf-8')).hexdigest()[:8], 16) % (2**31)


def _summarize_run_results(run_results):
    """Build summary statistics for a list of per-run dictionaries."""
    igd_vals = [r['igd'] for r in run_results]
    hv_vals = [r['hv'] for r in run_results]

    return {
        'igd_median': float(np.median(igd_vals)) if igd_vals else np.inf,
        'igd_iqr':    float(np.subtract(*np.percentile(igd_vals, [75, 25])))
                      if len(igd_vals) > 1 else 0.0,
        'hv_median':  float(np.median(hv_vals)),
        'hv_iqr':     float(np.subtract(*np.percentile(hv_vals, [75, 25])))
                      if len(hv_vals) > 1 else 0.0,
        'n_runs_feasible': sum(r['n_feasible'] > 0 for r in run_results),
        'runs': run_results,
    }


def _get_mp_context():
    """Choose multiprocessing context with a Linux-friendly default."""
    method = os.environ.get('CTAEA_MP_START', '').strip().lower()
    if method in {'fork', 'spawn', 'forkserver'}:
        return mp.get_context(method)

    if os.name == 'posix':
        # Fast and robust for this NumPy-heavy codebase on Linux clusters.
        return mp.get_context('fork')

    return mp.get_context('spawn')


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
    """Compute HV, falling back to Monte Carlo for large m."""
    from Metrics.hv import hypervolume, hypervolume_monte_carlo
    m = F.shape[1]
    if m <= 5:
        try:
            return hypervolume(F, ref_point)
        except Exception:
            pass
    return hypervolume_monte_carlo(F, ref_point, n_samples=50000)


def run_experiment(problem_names=None, m_values=None, algo_names=None,
    n_runs=51, output_dir='Results', verbose=False, quick_test=False,
    n_jobs=1, parallel_mode='per-config', gpu_ids=None, worker_threads=1):
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
    n_jobs        : int   number of parallel worker processes
    parallel_mode : str   'per-config' or 'global'
    gpu_ids       : list[str|int]|str  optional GPU IDs for round-robin task pinning
    worker_threads: int   BLAS/OpenMP threads per process (recommended 1 when n_jobs > 1)
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

    if n_jobs is None or n_jobs < 1:
        n_jobs = 1

    if parallel_mode not in {'per-config', 'global'}:
        raise ValueError("parallel_mode must be 'per-config' or 'global'")

    gpu_ids = _parse_gpu_ids(gpu_ids)

    # Prevent oversubscription when running many worker processes.
    if n_jobs > 1:
        threads = max(1, int(worker_threads))
        os.environ['OMP_NUM_THREADS'] = str(threads)
        os.environ['OPENBLAS_NUM_THREADS'] = str(threads)
        os.environ['MKL_NUM_THREADS'] = str(threads)
        os.environ['NUMEXPR_NUM_THREADS'] = str(threads)

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    results = {}

    total = len(problem_names) * len(m_values) * len(algo_names) * n_runs
    done = 0

    for prob_name in problem_names:
        results[prob_name] = {}
        for m in m_values:
            results[prob_name][m] = {}

    if parallel_mode == 'global' and n_jobs > 1:
        print(f"\nRunning in global parallel mode with {n_jobs} workers")
        if gpu_ids:
            print(f"GPU round-robin enabled across IDs: {gpu_ids}")

        configs = []
        all_tasks = []
        task_counter = 0

        for prob_name in problem_names:
            for m in m_values:
                for algo_name in algo_names:
                    configs.append((prob_name, m, algo_name))
                    for run_idx in range(n_runs):
                        seed = _stable_seed(prob_name, m, algo_name, run_idx)
                        gpu_id = gpu_ids[task_counter % len(gpu_ids)] if gpu_ids else None
                        all_tasks.append((run_idx, prob_name, m, algo_name, seed, verbose, gpu_id))
                        task_counter += 1

        run_store = {
            (prob_name, m, algo_name): [None] * n_runs
            for (prob_name, m, algo_name) in configs
        }

        ctx = _get_mp_context()
        with ProcessPoolExecutor(max_workers=n_jobs, mp_context=ctx) as ex:
            future_to_meta = {}
            for task in all_tasks:
                fut = ex.submit(_run_single_safe, task)
                future_to_meta[fut] = (task[0], task[1], task[2], task[3])

            for fut in as_completed(future_to_meta):
                run_idx, prob_name, m, algo_name = future_to_meta[fut]
                out_run_idx, res, err = fut.result()
                assert out_run_idx == run_idx

                if err is None:
                    run_store[(prob_name, m, algo_name)][run_idx] = res
                else:
                    run_store[(prob_name, m, algo_name)][run_idx] = {
                        'igd': np.inf,
                        'hv': 0.0,
                        'n_feasible': 0,
                        'time_sec': 0.0,
                        'F_nd': [],
                    }
                    print(f"Run failed | {prob_name} m={m} {algo_name} run={run_idx+1}: {err}")

                done += 1
                if done % 25 == 0 or done == total:
                    print(f"Progress: {done}/{total} runs completed")

        for prob_name, m, algo_name in configs:
            print(f"\n{'='*60}")
            print(f"Problem: {prob_name}, m={m}, Algorithm: {algo_name}")
            print(f"{'='*60}")

            run_results = run_store[(prob_name, m, algo_name)]
            summary = _summarize_run_results(run_results)

            print(f"  SUMMARY -> IGD: {summary['igd_median']:.4e} "
                  f"({summary['igd_iqr']:.2e}) | "
                  f"HV: {summary['hv_median']:.4e} "
                  f"({summary['hv_iqr']:.2e})")

            results[prob_name][m][algo_name] = summary

            save_path = out_dir / f"{prob_name}_m{m}_{algo_name.replace('/', '_')}.json"
            with open(save_path, 'w') as f:
                json.dump(_to_serializable(summary), f, indent=2)

    else:
        for prob_name in problem_names:
            for m in m_values:
                for algo_name in algo_names:
                    print(f"\n{'='*60}")
                    print(f"Problem: {prob_name}, m={m}, Algorithm: {algo_name}")
                    print(f"{'='*60}")

                    run_results = []
                    tasks = []
                    for run_idx in range(n_runs):
                        seed = _stable_seed(prob_name, m, algo_name, run_idx)
                        gpu_id = gpu_ids[run_idx % len(gpu_ids)] if gpu_ids else None
                        tasks.append((run_idx, prob_name, m, algo_name, seed, verbose, gpu_id))

                    if n_jobs <= 1:
                        ordered = []
                        for task in tasks:
                            run_idx, res, err = _run_single_safe(task)
                            ordered.append((run_idx, res, err))
                    else:
                        ordered = []
                        ctx = _get_mp_context()
                        with ProcessPoolExecutor(max_workers=n_jobs, mp_context=ctx) as ex:
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

                    summary = _summarize_run_results(run_results)

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


