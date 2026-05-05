"""
Main entry point for C-TAEA replication.

Usage:
    # Fast smoke test (~seconds–1 min): C-TAEA only, reduced FE budget
    python main.py --quick

    # Paper-style budget + compare vs peer algorithms (slow, tens of minutes):
    python main.py --quick --quick-paper --quick-peers

    # Run specific problem/algorithm:
    python main.py --problem C1DTLZ3 --m 3 --algo C-TAEA --runs 5

    # Full replication (51 runs, all problems, all m):
    python main.py --full

    # Analyze existing results:
    python main.py --analyze
"""
import sys
import os
import argparse
import numpy as np

# Ensure imports work
sys.path.insert(0, os.path.dirname(__file__))


def quick_demo(quick_paper_fe=False, quick_peers=False):
    """
    Demonstration on C1-DTLZ3 (3-objective): feasibility barrier vs peers (Fig. 4).

    By default this runs **only C-TAEA** with a **reduced** FE budget so it finishes
    quickly. Use ``--quick-paper`` for Table V multiplier and ``--quick-peers`` to
    include C-NSGA-III and C-MOEA/D (those are much slower).
    """
    print("="*60)
    print("C-TAEA Quick Demo: C1-DTLZ3, m=3")
    print("The hard problem where feasibility-driven methods fail")
    print("="*60)

    from Problems import C1DTLZ3
    from Algorithms import CTAEA, CNSGAIII, CMOEAD
    from Algorithms.utils import get_N_and_H
    from Experiments.runner import get_max_fe
    from Metrics.igd import igd
    import time

    m = 3
    problem = C1DTLZ3(n_obj=m)
    N, _ = get_N_and_H(m)
    if quick_paper_fe:
        max_fe = get_max_fe('C1DTLZ3', m, N)
    else:
        # Smoke test: enough generations to show behaviour without Table V runtime
        max_fe = 80 * N

    P_star = problem.get_pareto_front_reference(n_points=500)

    print(f"\nProblem: {problem.__class__.__name__}, "
          f"n_var={problem.n_var}, N={N}, max_FE={max_fe}")
    if not quick_paper_fe:
        print("(Reduced budget for speed. For paper Table V budget use: "
              "--quick --quick-paper)\n")
    else:
        print("(Paper Table V FE budget for C1-DTLZ3, m=3)\n")

    algos = [(CTAEA, 'C-TAEA')]
    if quick_peers:
        algos = [(CTAEA, 'C-TAEA'), (CNSGAIII, 'C-NSGA-III'), (CMOEAD, 'C-MOEA/D')]

    for AlgoClass, name in algos:
        t0 = time.time()
        algo = AlgoClass(problem=problem, N=N, max_fe=max_fe, seed=42, verbose=False)
        _, _, _ = algo.run()
        elapsed = time.time() - t0

        if hasattr(algo, 'get_nondominated'):
            _, F_nd, CV_nd = algo.get_nondominated()
        elif hasattr(algo, 'get_nondominated_CA'):
            _, F_nd, CV_nd = algo.get_nondominated_CA()
        else:
            from Algorithms.utils import non_dominated_indices
            feas = algo.pop_CV == 0
            F_f = algo.pop_F[feas]
            if len(F_f) > 0:
                nd = non_dominated_indices(F_f)
                F_nd = F_f[nd]
            else:
                F_nd = F_f

        n_feas = len(F_nd)
        igd_val = igd(F_nd, P_star) if n_feas > 0 else float('inf')

        print(f"  {name:15s}: feasible ND = {n_feas:3d}, "
              f"IGD = {igd_val:.4e}, time = {elapsed:.1f}s")
        if n_feas == 0:
            print(f"               (No feasible solutions found - "
                  f"algorithm trapped by infeasible barrier)")

    print("\nExpected with enough budget: C-TAEA finds feasible solutions (finite IGD).")
    if quick_peers:
        print("C-NSGA-III and C-MOEA/D typically find 0 feasible ND solutions here.")
    print("\n(Figure 4 / Table I in the paper use the full Table V FE budget.)\n")


def run_benchmark(args):
    """Run a specific benchmark configuration."""
    from Experiments.runner import (
        run_single, PROBLEM_CLASSES, ALGO_CLASSES, DEFAULT_CTAEA_METRIC_STAGES,
    )
    import json

    problem_name = args.problem
    m = args.m
    algo_name = args.algo
    n_runs = args.runs

    if problem_name not in PROBLEM_CLASSES:
        print(f"Unknown problem: {problem_name}")
        print(f"Available: {list(PROBLEM_CLASSES.keys())}")
        return

    if algo_name not in ALGO_CLASSES:
        print(f"Unknown algorithm: {algo_name}")
        print(f"Available: {list(ALGO_CLASSES.keys())}")
        return

    print(f"\nRunning: {algo_name} on {problem_name}, m={m}, {n_runs} runs")

    igd_vals, hv_vals = [], []
    cms = DEFAULT_CTAEA_METRIC_STAGES if (
        getattr(args, 'trace_metrics', False) and algo_name == 'C-TAEA') else None
    for run_idx in range(n_runs):
        seed = run_idx * 100
        res = run_single(
            problem_name, m, algo_name, seed=seed, verbose=args.verbose,
            collect_metrics_stages=cms,
        )
        igd_vals.append(res['igd'])
        hv_vals.append(res['hv'])
        print(f"  Run {run_idx+1:2d}: IGD={res['igd']:.4e}, "
              f"HV={res['hv']:.4e}, "
              f"Feasible={res['n_feasible']}, "
              f"Time={res['time_sec']:.1f}s")

    igd_valid = [v for v in igd_vals if not (np.isinf(v) or np.isnan(v))]
    print(f"\nSummary ({n_runs} runs):")
    print(f"  IGD: median={np.median(igd_valid):.4e}, "
          f"IQR={np.subtract(*np.percentile(igd_valid, [75, 25])):.2e}"
          if igd_valid else "  IGD: No feasible results")
    print(f"  HV:  median={np.median(hv_vals):.4e}, "
          f"IQR={np.subtract(*np.percentile(hv_vals, [75, 25])):.2e}")


def main():
    parser = argparse.ArgumentParser(
        description='C-TAEA replication of Li et al. (2019)')
    parser.add_argument('--quick', action='store_true',
                        help='Fast demo on C1-DTLZ3 (C-TAEA only, reduced FE budget)')
    parser.add_argument('--quick-paper', action='store_true',
                        help='With --quick: use paper Table V FE budget (much slower)')
    parser.add_argument('--quick-peers', action='store_true',
                        help='With --quick: also run C-NSGA-III and C-MOEA/D (slow)')
    parser.add_argument('--full', action='store_true',
                        help='Full replication (51 runs, all problems)')
    parser.add_argument('--analyze', action='store_true',
                        help='Analyze existing results in Results/')
    parser.add_argument('--problem', type=str, default='C1DTLZ3',
                        help='Problem name')
    parser.add_argument('--m', type=int, default=3,
                        help='Number of objectives')
    parser.add_argument('--algo', type=str, default='C-TAEA',
                        help='Algorithm name')
    parser.add_argument('--runs', type=int, default=5,
                        help='Number of independent runs')
    parser.add_argument('--verbose', action='store_true',
                        help='Verbose output')
    parser.add_argument('--output', type=str, default='Results',
                        help='Output directory')
    parser.add_argument('--workers', type=int, default=None,
                        help='Number of parallel workers (default: auto, recommend 48-60 for 64-thread systems)')
    parser.add_argument('--legacy', action='store_true',
                        help='Use legacy sequential runner instead of parallel runner')
    parser.add_argument('--trace-metrics', action='store_true',
                        help='For C-TAEA: store IGD/HV at 10%%–100%% of FE in results JSON')

    args = parser.parse_args()

    if args.quick:
        quick_demo(
            quick_paper_fe=args.quick_paper,
            quick_peers=args.quick_peers,
        )
    
    elif args.full:
        trace = True if args.trace_metrics else None
        # Use parallel runner by default for full replication
        if args.legacy:
            from Experiments.runner import run_experiment
            run_experiment(
                n_runs=51,
                output_dir=args.output,
                verbose=args.verbose,
                n_jobs=args.workers or 1,
                collect_metrics_stages=trace,
            )
        else:
            # Use highly parallel runner with spawn method
            import multiprocessing as mp
            try:
                mp.set_start_method('spawn', force=True)
            except RuntimeError:
                pass
            
            from Experiments.parallel_runner import run_experiment_parallel
            
            n_workers = args.workers
            if n_workers is None:
                cpu_count = mp.cpu_count()
                n_workers = min(60, max(1, cpu_count - 4))  # Leave headroom
            
            print(f"Running full replication with {n_workers} parallel workers (spawn method)...")
            run_experiment_parallel(
                n_runs=51,
                n_workers=n_workers,
                output_dir=args.output,
                verbose=True,
                collect_metrics_stages=trace,
            )
    
    elif args.analyze:
        from Analysis.analysis import load_and_analyze
        load_and_analyze(args.output)
    
    else:
        # Default: run specified problem/algo
        if len(sys.argv) == 1:
            # No args -> same fast demo as --quick
            quick_demo(quick_paper_fe=False, quick_peers=False)
        else:
            run_benchmark(args)


if __name__ == '__main__':
    main()
