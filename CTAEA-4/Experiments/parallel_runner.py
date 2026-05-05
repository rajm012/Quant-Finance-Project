"""
Highly Parallel Experiment Runner for C-TAEA Replication.
Optimized for multi-GPU servers with 64+ CPU threads.

Usage:
    # Run full replication with maximum parallelism
    python -m Experiments.parallel_runner --full --workers 60

    # Run specific problem/algorithm with parallel runs
    python -m Experiments.parallel_runner --problem C1DTLZ3 --m 3 --algo C-TAEA --workers 48
"""

import sys
import os
import numpy as np
import json
import time
import hashlib
import argparse
import warnings
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp

# Suppress numpy warnings
warnings.filterwarnings('ignore', category=RuntimeWarning)

# Add project root to path (needed for imports in main process)
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

# ─────────────────────────────────────────────────────────────────────────────
# Data Classes (must be module-level for pickling)
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class TaskConfig:
    """Configuration for a single experimental run."""
    problem_name: str
    m: int
    algo_name: str
    run_idx: int
    seed: int

@dataclass
class TaskResult:
    """Result from a single experimental run."""
    task: TaskConfig
    igd: float
    hv: float
    n_feasible: int
    time_sec: float
    F_nd: list
    error: Optional[str] = None

# ─────────────────────────────────────────────────────────────────────────────
# Configuration Tables (module-level for pickling)
# ─────────────────────────────────────────────────────────────────────────────

FE_TABLE = {
    'C1DTLZ1':  {3: 500,  5: 600,  8: 800,  10: 1000, 15: 1500},
    'C1DTLZ3':  {3: 1000, 5: 1500, 8: 2500, 10: 3500, 15: 5000},
    'C2DTLZ2':  {3: 250,  5: 350,  8: 500,  10: 750,  15: 1000},
    'C3DTLZ1':  {3: 750,  5: 1250, 8: 2000, 10: 3000, 15: 4000},
    'C3DTLZ4':  {3: 750,  5: 1250, 8: 2000, 10: 3000, 15: 4000},
    'DC1DTLZ1': {3: 500,  5: 600,  8: 800,  10: 1000, 15: 1500},
    'DC1DTLZ3': {3: 1000, 5: 1500, 8: 2500, 10: 3500, 15: 5000},
    'DC2DTLZ1': {3: 500,  5: 600,  8: 800,  10: 1000, 15: 1500},
    'DC2DTLZ3': {3: 1000, 5: 1500, 8: 2500, 10: 3500, 15: 5000},
    'DC3DTLZ1': {3: 750,  5: 1250, 8: 2000, 10: 3000, 15: 4000},
    'DC3DTLZ3': {3: 750,  5: 1250, 8: 2000, 10: 3000, 15: 4000},
}

ZR_TABLE = {'C3DTLZ4': 2.1, 'default': 1.1}

# ─────────────────────────────────────────────────────────────────────────────
# Worker Function (MODULE-LEVEL - CRITICAL FOR PICKLING)
# ─────────────────────────────────────────────────────────────────────────────

def run_single_task(task: TaskConfig) -> TaskResult:
    """
    Execute a single experimental run in isolated process.
    
    This function is MODULE-LEVEL so it can be pickled by multiprocessing.
    All imports are done INSIDE the function to ensure fresh state in subprocess.
    """
    # Re-import everything inside the worker function
    import sys
    import os
    import numpy as np
    import time
    import warnings
    
    warnings.filterwarnings('ignore')
    
    # Add project root to path
    project_root = os.path.dirname(os.path.dirname(__file__))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    
    # Imports must be inside function for spawn method
    from Problems import (
        C1DTLZ1, C1DTLZ3, C2DTLZ2, C3DTLZ1, C3DTLZ4,
        DC1DTLZ1, DC1DTLZ3, DC2DTLZ1, DC2DTLZ3, DC3DTLZ1, DC3DTLZ3,
    )
    from Algorithms import CTAEA, CMOEAD, CNSGAIII, CMOEAD_DD, IDBEA, CMOEA
    from Algorithms.utils import get_N_and_H, non_dominated_indices
    from Metrics.igd import igd
    from Metrics.hv import hypervolume, hypervolume_monte_carlo
    
    # Local copies of tables (module-level constants not available in subprocess)
    FE_TABLE_LOCAL = {
        'C1DTLZ1':  {3: 500,  5: 600,  8: 800,  10: 1000, 15: 1500},
        'C1DTLZ3':  {3: 1000, 5: 1500, 8: 2500, 10: 3500, 15: 5000},
        'C2DTLZ2':  {3: 250,  5: 350,  8: 500,  10: 750,  15: 1000},
        'C3DTLZ1':  {3: 750,  5: 1250, 8: 2000, 10: 3000, 15: 4000},
        'C3DTLZ4':  {3: 750,  5: 1250, 8: 2000, 10: 3000, 15: 4000},
        'DC1DTLZ1': {3: 500,  5: 600,  8: 800,  10: 1000, 15: 1500},
        'DC1DTLZ3': {3: 1000, 5: 1500, 8: 2500, 10: 3500, 15: 5000},
        'DC2DTLZ1': {3: 500,  5: 600,  8: 800,  10: 1000, 15: 1500},
        'DC2DTLZ3': {3: 1000, 5: 1500, 8: 2500, 10: 3500, 15: 5000},
        'DC3DTLZ1': {3: 750,  5: 1250, 8: 2000, 10: 3000, 15: 4000},
        'DC3DTLZ3': {3: 750,  5: 1250, 8: 2000, 10: 3000, 15: 4000},
    }
    ZR_TABLE_LOCAL = {'C3DTLZ4': 2.1, 'default': 1.1}
    
    PROBLEM_CLASSES = {
        'C1DTLZ1': C1DTLZ1, 'C1DTLZ3': C1DTLZ3, 'C2DTLZ2': C2DTLZ2,
        'C3DTLZ1': C3DTLZ1, 'C3DTLZ4': C3DTLZ4,
        'DC1DTLZ1': DC1DTLZ1, 'DC1DTLZ3': DC1DTLZ3,
        'DC2DTLZ1': DC2DTLZ1, 'DC2DTLZ3': DC2DTLZ3,
        'DC3DTLZ1': DC3DTLZ1, 'DC3DTLZ3': DC3DTLZ3,
    }
    ALGO_CLASSES = {
        'C-TAEA': CTAEA, 'C-NSGA-III': CNSGAIII, 'C-MOEA/D': CMOEAD,
        'C-MOEA/DD': CMOEAD_DD, 'I-DBEA': IDBEA, 'CMOEA': CMOEA,
    }
    
    try:
        ProbClass = PROBLEM_CLASSES[task.problem_name]
        AlgoClass = ALGO_CLASSES[task.algo_name]
        
        problem = ProbClass(n_obj=task.m)
        N, _ = get_N_and_H(task.m)
        mult = FE_TABLE_LOCAL.get(task.problem_name, {}).get(task.m, 500)
        max_fe = mult * N
        
        algo = AlgoClass(
            problem=problem,
            N=N,
            max_fe=max_fe,
            seed=task.seed,
            verbose=False,
        )
        
        t0 = time.time()
        algo.run()
        elapsed = time.time() - t0
        
        # Extract non-dominated feasible
        if hasattr(algo, 'get_nondominated'):
            X_nd, F_nd, CV_nd = algo.get_nondominated()
        elif hasattr(algo, 'get_nondominated_CA'):
            X_nd, F_nd, CV_nd = algo.get_nondominated_CA()
        else:
            feas = algo.pop_CV == 0
            X_f = algo.pop_X[feas]
            F_f = algo.pop_F[feas]
            if len(F_f) == 0:
                X_nd, F_nd = X_f, F_f
            else:
                nd = non_dominated_indices(F_f)
                X_nd, F_nd = X_f[nd], F_f[nd]
        
        # Compute metrics
        P_star = problem.get_pareto_front_reference(n_points=500)
        igd_val = igd(F_nd, P_star) if len(F_nd) > 0 else np.inf
        
        val = ZR_TABLE_LOCAL.get(task.problem_name, ZR_TABLE_LOCAL['default'])
        ref_point = np.full(task.m, val)
        
        def _compute_hv(F, ref):
            """Compute HV with adaptive sampling (Optimization #4)."""
            m = F.shape[1]
            n_points = len(F)
            
            # Exact HV for low dimensions (paper standard)
            if m <= 5:
                try:
                    return hypervolume(F, ref)
                except Exception:
                    pass
            
            # Adaptive sampling for high dimensions
            n_samples = min(50000, max(15000, n_points * 150))
            if m > 12:
                n_samples = min(n_samples, 25000)
            
            return hypervolume_monte_carlo(F, ref, n_samples=n_samples)
        
        hv_val = _compute_hv(F_nd, ref_point) if len(F_nd) > 0 else 0.0
        
        return TaskResult(
            task=task,
            igd=float(igd_val),
            hv=float(hv_val),
            n_feasible=int(len(F_nd)),
            time_sec=float(elapsed),
            F_nd=F_nd.tolist() if len(F_nd) > 0 else [],
            error=None
        )
        
    except Exception as e:
        import traceback
        return TaskResult(
            task=task,
            igd=np.inf,
            hv=0.0,
            n_feasible=0,
            time_sec=0.0,
            F_nd=[],
            error=f"{str(e)}\n{traceback.format_exc()[:500]}"
        )

# ─────────────────────────────────────────────────────────────────────────────
# Main Execution Function
# ─────────────────────────────────────────────────────────────────────────────

def run_experiment_parallel(
    problem_names=None,
    m_values=None,
    algo_names=None,
    n_runs=51,
    n_workers=48,
    output_dir='Results',
    verbose=True,
    quick_test=False,
):
    """
    Run full experiment with maximum parallelization using ProcessPoolExecutor.
    """
    if quick_test:
        n_runs = 3
        m_values = [3] if m_values is None else m_values[:1]
    
    if problem_names is None:
        from Problems import (
            C1DTLZ1, C1DTLZ3, C2DTLZ2, C3DTLZ1, C3DTLZ4,
            DC1DTLZ1, DC1DTLZ3, DC2DTLZ1, DC2DTLZ3, DC3DTLZ1, DC3DTLZ3,
        )
        problem_names = ['C1DTLZ1', 'C1DTLZ3', 'C2DTLZ2', 'C3DTLZ1', 'C3DTLZ4',
                        'DC1DTLZ1', 'DC1DTLZ3', 'DC2DTLZ1', 'DC2DTLZ3', 'DC3DTLZ1', 'DC3DTLZ3']
    if m_values is None:
        m_values = [3, 5, 8, 10, 15]
    if algo_names is None:
        algo_names = ['C-TAEA', 'C-NSGA-III', 'C-MOEA/D', 'C-MOEA/DD', 'I-DBEA', 'CMOEA']
    
    # Generate all tasks
    tasks = []
    for prob_name in problem_names:
        for m in m_values:
            seed_key = f"{prob_name}|{m}"
            base_seed = int(hashlib.sha256(seed_key.encode("utf-8")).hexdigest()[:8], 16)
            for algo_name in algo_names:
                for run_idx in range(n_runs):
                    seed = (base_seed + run_idx * 100) % (2**31)
                    tasks.append(TaskConfig(prob_name, m, algo_name, run_idx, seed))
    
    total_tasks = len(tasks)
    
    if verbose:
        print(f"\n{'#'*70}")
        print(f"# C-TAEA Parallel Experiment Runner")
        print(f"# Total tasks: {total_tasks}")
        print(f"# Problems: {problem_names}")
        print(f"# M-values: {m_values}")
        print(f"# Algorithms: {algo_names}")
        print(f"# Runs per config: {n_runs}")
        print(f"# Workers: {n_workers}")
        print(f"{'#'*70}\n")
    
    # Run in parallel using ProcessPoolExecutor with spawn
    results = []
    completed = 0
    failed = 0
    
    # Limit workers
    n_workers = min(n_workers, total_tasks, 60)
    
    if verbose:
        print(f"Starting parallel execution with {n_workers} workers (spawn method)...")
        # Update every 1% or every 10 tasks, whichever is smaller
        progress_interval = max(1, min(total_tasks // 100, 10))
        print(f"Progress updates every {progress_interval} tasks\n")
    
    start_time = time.time()
    last_progress_time = start_time
    
    # Use spawn context explicitly
    ctx = mp.get_context('spawn')
    
    with ProcessPoolExecutor(max_workers=n_workers, mp_context=ctx) as executor:
        # Submit all tasks
        future_to_idx = {executor.submit(run_single_task, task): i 
                        for i, task in enumerate(tasks)}
        
        # Collect results
        for future in as_completed(future_to_idx):
            idx = future_to_idx[future]
            try:
                result = future.result(timeout=3600)  # 1 hour timeout
                results.append(result)
                completed += 1
                if result.error:
                    failed += 1
                    if verbose:
                        print(f"\n  ERROR in {result.task.problem_name} m={result.task.m} "
                              f"{result.task.algo_name} run={result.task.run_idx}: {result.error[:100]}")
            except Exception as e:
                failed += 1
                completed += 1
                task = tasks[idx]
                if verbose:
                    print(f"\n  ERROR in {task.problem_name} m={task.m} "
                          f"{task.algo_name} run={task.run_idx}: {e}")
                results.append(TaskResult(
                    task=task, igd=np.inf, hv=0.0, n_feasible=0,
                    time_sec=0.0, F_nd=[], error=str(e)
                ))
            
            # Progress update - every N tasks OR every 30 seconds
            current_time = time.time()
            time_since_last = current_time - last_progress_time
            progress_interval = max(1, min(total_tasks // 100, 10))
            
            if verbose and (completed % progress_interval == 0 or completed == total_tasks or time_since_last > 30):
                elapsed = current_time - start_time
                rate = completed / elapsed if elapsed > 0 else 0
                eta = (total_tasks - completed) / rate if rate > 0 else 0
                print(f"  [{completed:4d}/{total_tasks:4d}] {100*completed/total_tasks:5.1f}% | "
                      f"Rate: {rate:5.2f} tasks/s | ETA: {eta/60:5.1f}min | Failed: {failed}", flush=True)
                last_progress_time = current_time
    
    total_elapsed = time.time() - start_time
    
    if verbose:
        print(f"\nCompleted {completed}/{total_tasks} tasks in {total_elapsed/60:.1f} minutes")
        print(f"Success rate: {100*(completed-failed)/max(completed,1):.1f}%")
    
    # Save results
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Group by config
    grouped = {}
    for res in results:
        key = (res.task.problem_name, res.task.m, res.task.algo_name)
        if key not in grouped:
            grouped[key] = []
        grouped[key].append(res)
    
    # Save individual files
    for (prob, m, algo), group in grouped.items():
        safe_algo = algo.replace('/', '_')
        save_path = out_dir / f"{prob}_m{m}_{safe_algo}.json"
        
        igd_vals = [r.igd for r in group]
        hv_vals = [r.hv for r in group]
        igd_valid = [v for v in igd_vals if not (np.isinf(v) or np.isnan(v))]
        
        data = {
            'problem': prob, 'm': m, 'algorithm': algo,
            'igd_median': float(np.median(igd_valid)) if igd_valid else np.inf,
            'igd_iqr': float(np.subtract(*np.percentile(igd_valid, [75, 25]))) if len(igd_valid) > 1 else 0.0,
            'hv_median': float(np.median(hv_vals)),
            'hv_iqr': float(np.subtract(*np.percentile(hv_vals, [75, 25]))) if len(hv_vals) > 1 else 0.0,
            'n_runs_feasible': sum(1 for r in group if r.n_feasible > 0),
            'n_runs_total': len(group),
            'runs': [{'run_idx': r.task.run_idx, 'igd': r.igd, 'hv': r.hv,
                     'n_feasible': r.n_feasible, 'time_sec': r.time_sec} for r in group]
        }
        
        with open(save_path, 'w') as f:
            json.dump(data, f, indent=2)
    
    # Save combined
    summary = {}
    for (prob, m, algo), group in grouped.items():
        if prob not in summary:
            summary[prob] = {}
        if m not in summary[prob]:
            summary[prob][m] = {}
        
        igd_vals = [r.igd for r in group]
        hv_vals = [r.hv for r in group]
        igd_valid = [v for v in igd_vals if not (np.isinf(v) or np.isnan(v))]
        
        summary[prob][m][algo] = {
            'igd_median': float(np.median(igd_valid)) if igd_valid else np.inf,
            'igd_iqr': float(np.subtract(*np.percentile(igd_valid, [75, 25]))) if len(igd_valid) > 1 else 0.0,
            'hv_median': float(np.median(hv_vals)),
            'hv_iqr': float(np.subtract(*np.percentile(hv_vals, [75, 25]))) if len(hv_vals) > 1 else 0.0,
            'n_runs_feasible': sum(1 for r in group if r.n_feasible > 0),
            'n_runs_total': len(group),
        }
    
    with open(out_dir / 'all_results.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    if verbose:
        print(f"\nResults saved to {out_dir}/")
        print(f"  - {len(grouped)} individual files")
        print(f"  - 1 combined file")
    
    return summary


def main():
    parser = argparse.ArgumentParser(description='C-TAEA Parallel Runner')
    parser.add_argument('--full', action='store_true', help='Full replication')
    parser.add_argument('--quick', action='store_true', help='Quick test')
    parser.add_argument('--problem', type=str)
    parser.add_argument('--problems', nargs='+')
    parser.add_argument('--m', type=int)
    parser.add_argument('--m-values', nargs='+', type=int)
    parser.add_argument('--algo', type=str)
    parser.add_argument('--algos', nargs='+')
    parser.add_argument('--runs', type=int, default=51)
    parser.add_argument('--workers', type=int, default=48)
    parser.add_argument('--output', type=str, default='Results')
    
    args = parser.parse_args()
    
    # Set spawn method at main entry point
    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        pass
    
    if args.quick:
        run_experiment_parallel(quick_test=True, n_workers=args.workers, output_dir=args.output)
    elif args.full:
        run_experiment_parallel(n_workers=args.workers, output_dir=args.output)
    else:
        probs = args.problems or ([args.problem] if args.problem else None)
        ms = args.m_values or ([args.m] if args.m else None)
        algos = args.algos or ([args.algo] if args.algo else None)
        run_experiment_parallel(
            problem_names=probs, m_values=ms, algo_names=algos,
            n_runs=args.runs, n_workers=args.workers, output_dir=args.output
        )


if __name__ == '__main__':
    main()
