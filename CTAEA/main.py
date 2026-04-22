"""
Main entry point for C-TAEA replication.

Usage:
    # Quick test (3 runs, m=3 only, C-DTLZ):
    python main.py --quick

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


def quick_demo():
    """
    Quick demonstration: run C-TAEA on C1-DTLZ3 (3-objective),
    the problem where all peer algorithms fail (Fig. 4 in paper).
    """
    print("="*60)
    print("C-TAEA Quick Demo: C1-DTLZ3, m=3")
    print("The hard problem where feasibility-driven methods fail")
    print("="*60)

    from Problems import C1DTLZ3
    from Algorithms import CTAEA, CNSGAIII, CMOEAD
    from Algorithms.utils import get_N_and_H
    from Metrics.igd import igd

    m = 3
    problem = C1DTLZ3(n_obj=m)
    N, _ = get_N_and_H(m)
    max_fe = 1000 * N  # from Table V

    P_star = problem.get_pareto_front_reference(n_points=500)

    print(f"\nProblem: {problem.__class__.__name__}, "
          f"n_var={problem.n_var}, N={N}, max_FE={max_fe}\n")

    for AlgoClass, name in [(CTAEA, 'C-TAEA'), (CNSGAIII, 'C-NSGA-III'), (CMOEAD, 'C-MOEA/D')]:
        algo = AlgoClass(problem=problem, N=N, max_fe=max_fe, seed=42, verbose=False)
        _, _, _ = algo.run()

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

        print(f"  {name:15s}: feasible solutions = {n_feas:3d}, "
              f"IGD = {igd_val:.4e}")
        if n_feas == 0:
            print(f"               (No feasible solutions found - "
                  f"algorithm trapped by infeasible barrier)")

    print("\nExpected: C-TAEA finds feasible solutions (IGD > 0).")
    print("C-NSGA-III and C-MOEA/D should find 0 feasible solutions (IGD=inf).")
    print("\nThis matches Figure 4 and Table I in the paper.\n")


def run_benchmark(args):
    """Run a specific benchmark configuration."""
    from Experiments.runner import run_single, PROBLEM_CLASSES, ALGO_CLASSES
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
    for run_idx in range(n_runs):
        seed = run_idx * 100
        res = run_single(problem_name, m, algo_name, seed=seed, verbose=args.verbose)
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
                        help='Quick demo on C1-DTLZ3')
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

    args = parser.parse_args()

    if args.quick:
        quick_demo()
    
    elif args.full:
        from Experiments.runner import run_experiment
        run_experiment(
            n_runs=51,
            output_dir=args.output,
            verbose=args.verbose,
        )
    
    elif args.analyze:
        from Analysis.analysis import load_and_analyze
        load_and_analyze(args.output)
    
    else:
        # Default: run specified problem/algo
        if len(sys.argv) == 1:
            # No args -> quick demo
            quick_demo()
        else:
            run_benchmark(args)


if __name__ == '__main__':
    main()
