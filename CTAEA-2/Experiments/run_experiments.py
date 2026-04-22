"""
Experiment Runner
=================
Reproduces the experimental setup from the C-TAEA paper:
  - 51 independent runs per algorithm-problem-m combination
  - Records IGD and HV values
  - Saves results to Results/ directory

Usage:
    python Experiments/run_experiments.py
    python Experiments/run_experiments.py --problem C1-DTLZ3 --m 3 --runs 5
    python Experiments/run_experiments.py --algorithm C-TAEA --quick
"""

import numpy as np
import os, sys, json, argparse, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Problems import ALL_PROBLEMS, C_DTLZ_PROBLEMS, DC_DTLZ_PROBLEMS
from Algorithms import ALL_ALGORITHMS
from Metrics import igd, hypervolume, get_reference_points, get_reference_point_hv
from utils import get_population_size, get_fe_budget


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

N_RUNS = 51          # Paper uses 51 independent runs
OBJECTIVES = [3, 5, 8, 10, 15]

# All problems from the paper
ALL_PROBLEM_NAMES = C_DTLZ_PROBLEMS + DC_DTLZ_PROBLEMS

# All algorithms
ALL_ALGO_NAMES = ['C-TAEA', 'C-NSGA-III', 'C-MOEA/D', 'C-MOEA/DD', 'I-DBEA', 'CMOEA']


# ---------------------------------------------------------------------------
# Single run
# ---------------------------------------------------------------------------

def run_single(algo_name: str, problem_name: str, m: int,
               seed: int = None, verbose: bool = False) -> dict:
    """
    Run a single algorithm on a single problem instance.
    Returns dict with IGD and HV values.
    """
    # Instantiate problem
    ProbClass = ALL_PROBLEMS[problem_name]
    problem = ProbClass(m=m)

    N = get_population_size(m)
    max_fe = get_fe_budget(problem_name, m)

    # Instantiate algorithm
    AlgoClass = ALL_ALGORITHMS[algo_name]
    algo = AlgoClass(
        problem=problem,
        m=m,
        N=N,
        max_fe=max_fe,
        seed=seed,
        verbose=verbose
    )

    t0 = time.time()
    if algo_name == 'C-TAEA':
        CA, DA = algo.run()
        P = algo.get_pareto_front('CA')
        if len(P) == 0:
            P = algo.get_pareto_front('combined')
    else:
        algo.run()
        P = algo.get_pareto_front()

    elapsed = time.time() - t0

    # Compute metrics
    P_star = get_reference_points(problem_name, m, n_points=500)
    z_r    = get_reference_point_hv(problem_name, m)

    if len(P) > 0:
        igd_val = igd(P, P_star)
        # Remove solutions dominated by z_r for HV
        dominated_by_ref = np.all(P < z_r, axis=1)
        P_hv = P[dominated_by_ref]
        hv_val = hypervolume(P_hv, z_r) if len(P_hv) > 0 else 0.0
    else:
        igd_val = np.inf
        hv_val  = 0.0

    return {
        'igd': igd_val,
        'hv':  hv_val,
        'n_feasible': len(P),
        'time': elapsed,
        'seed': seed,
    }


# ---------------------------------------------------------------------------
# Full experiment for one (algo, problem, m)
# ---------------------------------------------------------------------------

def run_experiment(algo_name: str, problem_name: str, m: int,
                   n_runs: int = N_RUNS,
                   results_dir: str = 'Results',
                   verbose: bool = False) -> dict:
    """
    Run n_runs independent trials and compute statistics.
    """
    os.makedirs(results_dir, exist_ok=True)

    result_file = os.path.join(
        results_dir,
        f"{algo_name.replace('/', '_')}_{problem_name.replace('-', '')}_{m}obj.json"
    )

    # Check if already computed
    if os.path.exists(result_file):
        with open(result_file) as f:
            return json.load(f)

    print(f"  Running {algo_name} on {problem_name} m={m} ({n_runs} runs)...", flush=True)

    igd_vals = []
    hv_vals  = []
    feasible_counts = []

    for run_idx in range(n_runs):
        seed = run_idx * 1000 + hash(algo_name + problem_name + str(m)) % 1000
        seed = abs(seed) % (2**31)
        try:
            res = run_single(algo_name, problem_name, m, seed=seed, verbose=False)
            igd_v = res['igd'] if not np.isinf(res['igd']) else np.nan
            igd_vals.append(igd_v)
            hv_vals.append(res['hv'])
            feasible_counts.append(res['n_feasible'])
        except Exception as e:
            print(f"    Warning: run {run_idx} failed: {e}")
            igd_vals.append(np.nan)
            hv_vals.append(0.0)
            feasible_counts.append(0)

        if (run_idx + 1) % 10 == 0:
            print(f"    Completed {run_idx+1}/{n_runs} runs", flush=True)

    # Compute statistics (median, IQR)
    igd_arr = np.array(igd_vals)
    hv_arr  = np.array(hv_vals)

    def safe_median(arr):
        v = arr[~np.isnan(arr)]
        return float(np.median(v)) if len(v) > 0 else np.nan

    def safe_iqr(arr):
        v = arr[~np.isnan(arr)]
        if len(v) == 0:
            return np.nan
        return float(np.percentile(v, 75) - np.percentile(v, 25))

    result = {
        'algorithm':    algo_name,
        'problem':      problem_name,
        'm':            m,
        'n_runs':       n_runs,
        'igd_median':   safe_median(igd_arr),
        'igd_iqr':      safe_iqr(igd_arr),
        'hv_median':    safe_median(hv_arr),
        'hv_iqr':       safe_iqr(hv_arr),
        'igd_all':      [float(v) if not np.isnan(v) else None for v in igd_vals],
        'hv_all':       [float(v) for v in hv_vals],
        'feasible_runs': int(np.sum(np.array(feasible_counts) > 0)),
        'n_feasible_all': [int(v) for v in feasible_counts],
    }

    with open(result_file, 'w') as f:
        json.dump(result, f, indent=2)

    print(f"    IGD median={result['igd_median']:.4e}, IQR={result['igd_iqr']:.4e} | "
          f"HV median={result['hv_median']:.4e} | "
          f"Feasible runs: {result['feasible_runs']}/{n_runs}")

    return result


# ---------------------------------------------------------------------------
# Full experiment table (matching Tables I and III in paper)
# ---------------------------------------------------------------------------

def run_full_experiment(algo_names: list = None,
                         problem_names: list = None,
                         objectives: list = None,
                         n_runs: int = N_RUNS,
                         results_dir: str = 'Results',
                         verbose: bool = False):
    """
    Run all experiments. Matches the full experimental setup of the paper.
    """
    if algo_names is None:
        algo_names = ALL_ALGO_NAMES
    if problem_names is None:
        problem_names = ALL_PROBLEM_NAMES
    if objectives is None:
        objectives = OBJECTIVES

    all_results = {}
    total = len(algo_names) * len(problem_names) * len(objectives)
    done = 0

    for algo_name in algo_names:
        for problem_name in problem_names:
            for m in objectives:
                done += 1
                print(f"\n[{done}/{total}] {algo_name} | {problem_name} | m={m}")
                key = f"{algo_name}_{problem_name}_m{m}"
                try:
                    result = run_experiment(
                        algo_name, problem_name, m,
                        n_runs=n_runs,
                        results_dir=results_dir,
                        verbose=verbose
                    )
                    all_results[key] = result
                except Exception as e:
                    print(f"  ERROR: {e}")
                    import traceback; traceback.print_exc()

    return all_results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run C-TAEA experiments')
    parser.add_argument('--algorithm', type=str, default=None,
                        help='Algorithm name (default: all)')
    parser.add_argument('--problem', type=str, default=None,
                        help='Problem name (default: all)')
    parser.add_argument('--m', type=int, default=None,
                        help='Number of objectives (default: all)')
    parser.add_argument('--runs', type=int, default=N_RUNS,
                        help=f'Number of runs (default: {N_RUNS})')
    parser.add_argument('--results_dir', type=str, default='Results',
                        help='Directory to save results')
    parser.add_argument('--quick', action='store_true',
                        help='Quick test: 3 objectives, 5 runs, C-TAEA only')
    parser.add_argument('--verbose', action='store_true')
    args = parser.parse_args()

    if args.quick:
        print("=== Quick test mode ===")
        run_full_experiment(
            algo_names=['C-TAEA'],
            problem_names=['C1-DTLZ1', 'C1-DTLZ3', 'C2-DTLZ2'],
            objectives=[3],
            n_runs=3,
            results_dir=args.results_dir
        )
    else:
        algos    = [args.algorithm] if args.algorithm else None
        problems = [args.problem]   if args.problem   else None
        objs     = [args.m]         if args.m         else None

        run_full_experiment(
            algo_names=algos,
            problem_names=problems,
            objectives=objs,
            n_runs=args.runs,
            results_dir=args.results_dir,
            verbose=args.verbose
        )
