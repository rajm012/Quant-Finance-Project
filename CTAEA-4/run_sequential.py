#!/usr/bin/env python3
"""
Sequential Model Runner - Runs 6 algorithms one at a time.
Ordered by computational cost: lightest first, heaviest last.

Usage:
    # Run all 6 models sequentially (recommended order)
    python run_sequential.py --all
    
    # Run specific model only
    python run_sequential.py --algo "C-TAEA"
    
    # Run with custom workers (for parallel runs within one model)
    python run_sequential.py --all --workers 16
"""

import sys
import os
import time
import json
import argparse
from pathlib import Path
from datetime import datetime

# ─────────────────────────────────────────────────────────────────────────────
# Configuration: 6 models ordered by computational cost (lightest → heaviest)
# ─────────────────────────────────────────────────────────────────────────────

MODEL_ORDER = [
    {'name': 'CMOEA',      'algo': 'CMOEA',      'complexity': 'Light',     'est_time_hours': 1.5},
    {'name': 'C-NSGA-III', 'algo': 'C-NSGA-III', 'complexity': 'Light',     'est_time_hours': 1.8},
    {'name': 'I-DBEA',     'algo': 'I-DBEA',     'complexity': 'Medium',    'est_time_hours': 2.0},
    {'name': 'C-MOEA/D',   'algo': 'C-MOEA/D',   'complexity': 'Medium+',   'est_time_hours': 2.2},
    {'name': 'C-MOEA/DD',  'algo': 'C-MOEA/DD',  'complexity': 'Heavy',     'est_time_hours': 2.5},
    {'name': 'C-TAEA',     'algo': 'C-TAEA',     'complexity': 'Heaviest',  'est_time_hours': 3.0},
]

ALL_PROBLEMS = ['C1DTLZ1', 'C1DTLZ3', 'C2DTLZ2', 'C3DTLZ1', 'C3DTLZ4',
                'DC1DTLZ1', 'DC1DTLZ3', 'DC2DTLZ1', 'DC2DTLZ3', 'DC3DTLZ1', 'DC3DTLZ3']
ALL_M_VALUES = [3, 5, 8, 10, 15]
N_RUNS = 51

# ─────────────────────────────────────────────────────────────────────────────
# Runner Functions
# ─────────────────────────────────────────────────────────────────────────────

def run_single_model(algo_name, problems, m_values, n_runs, workers, output_dir, resume=True):
    """
    Run one algorithm on all problem/m combinations.
    Uses parallel processing within the model for speed.
    """
    from Experiments.parallel_runner import run_experiment_parallel
    import multiprocessing as mp
    
    # Ensure spawn method
    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        pass
    
    print(f"\n{'='*70}")
    print(f"Running: {algo_name}")
    print(f"Problems: {problems}")
    print(f"M-values: {m_values}")
    print(f"Runs per config: {n_runs}")
    print(f"Parallel workers: {workers}")
    print(f"Output: {output_dir}")
    print(f"{'='*70}\n")
    
    start = time.time()
    
    try:
        results = run_experiment_parallel(
            problem_names=problems,
            m_values=m_values,
            algo_names=[algo_name],
            n_runs=n_runs,
            n_workers=workers,
            output_dir=output_dir,
            verbose=True,
        )
        
        elapsed = time.time() - start
        
        # Save completion marker
        marker_file = Path(output_dir) / f"_{algo_name.replace('/', '_')}_COMPLETE"
        with open(marker_file, 'w') as f:
            f.write(f"Completed: {datetime.now().isoformat()}\n")
            f.write(f"Elapsed: {elapsed/3600:.2f} hours\n")
        
        print(f"\n✓ {algo_name} completed in {elapsed/3600:.2f} hours")
        return True, elapsed, results
        
    except Exception as e:
        elapsed = time.time() - start
        print(f"\n✗ {algo_name} failed after {elapsed/3600:.2f} hours")
        print(f"Error: {e}")
        return False, elapsed, None


def check_completed(output_dir, algo_name):
    """Check if algorithm was already completed."""
    marker = Path(output_dir) / f"_{algo_name.replace('/', '_')}_COMPLETE"
    return marker.exists()


def print_progress_summary(completed, failed, total, times):
    """Print summary of progress so far."""
    print(f"\n{'#'*70}")
    print("# PROGRESS SUMMARY")
    print(f"{'#'*70}")
    print(f"Completed: {len(completed)}/{total}")
    print(f"Failed: {len(failed)}")
    print(f"Remaining: {total - len(completed) - len(failed)}")
    
    if times:
        avg_time = sum(times.values()) / len(times) / 3600
        total_time = sum(times.values()) / 3600
        print(f"\nAverage per model: {avg_time:.2f} hours")
        print(f"Total elapsed: {total_time:.2f} hours")
        
        remaining = [m['algo'] for m in model_order if m['algo'] not in completed]
        est_remaining = sum(m['est_time_hours'] for m in model_order 
                           if m['algo'] in remaining)
        print(f"Est. remaining: {est_remaining:.1f} hours")
    
    print(f"{'#'*70}\n")


def run_all_sequential(problems, m_values, n_runs, workers, output_dir, resume=True, model_order=None):
    """
    Run all models sequentially in order of computational cost.
    """
    if model_order is None:
        model_order = MODEL_ORDER
        
    print("\n" + "#"*70)
    print("# SEQUENTIAL MODEL EXECUTION")
    print("# Running models from lightest to heaviest")
    print("#"*70)
    
    print("\nExecution order:")
    for i, model in enumerate(model_order, 1):
        status = "[PENDING]"
        if check_completed(output_dir, model['algo']):
            status = "[DONE]"
        print(f"  {i}. {model['algo']:<12} ({model['complexity']:<8}) {status}")
    
    completed = []
    failed = []
    times = {}
    all_results = {}
    
    for i, model in enumerate(model_order, 1):
        algo = model['algo']
        
        # Check if already done (resume support)
        if resume and check_completed(output_dir, algo):
            print(f"\n[{i}/{len(model_order)}] {algo} already completed. Skipping...")
            completed.append(algo)
            continue
        
        print(f"\n\n[{i}/{len(model_order)}] Starting {algo} ({model['complexity']}, est: {model['est_time_hours']}h)")
        print("-" * 70)
        
        # Run the model
        success, elapsed, results = run_single_model(
            algo_name=algo,
            problems=problems,
            m_values=m_values,
            n_runs=n_runs,
            workers=workers,
            output_dir=output_dir,
            resume=resume,
        )
        
        times[algo] = elapsed
        all_results[algo] = results
        
        if success:
            completed.append(algo)
        else:
            failed.append(algo)
            print(f"\n⚠ {algo} failed. Continuing with next model...")
        
        # Print progress summary
        print_progress_summary(completed, failed, len(model_order), times)
        
        # Optional: pause between models for resource cleanup
        if i < len(model_order) and success:
            print("\nPausing 5 seconds for resource cleanup...")
            time.sleep(5)
    
    # Final summary
    print("\n" + "="*70)
    print("SEQUENTIAL EXECUTION COMPLETE")
    print("="*70)
    print(f"Completed: {len(completed)}/6 - {completed}")
    print(f"Failed: {len(failed)}/6 - {failed}")
    print(f"Total time: {sum(times.values())/3600:.2f} hours")
    
    # Save overall summary
    summary = {
        'completed': completed,
        'failed': failed,
        'times': {k: v/3600 for k, v in times.items()},
        'total_hours': sum(times.values()) / 3600,
    }
    
    with open(Path(output_dir) / '_sequential_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    return completed, failed, all_results


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Sequential Model Runner - runs 6 algorithms one at a time',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run all 6 models sequentially (recommended)
  python run_sequential.py --all --workers 16
  
  # Run only the lightest 2 models (for testing)
  python run_sequential.py --models 0,1 --workers 8
  
  # Run specific model
  python run_sequential.py --algo "C-TAEA" --workers 16
  
  # Quick test (3 runs only)
  python run_sequential.py --all --runs 3 --workers 4
  
  # Run with fewer problems for faster testing
  python run_sequential.py --all --problems C1DTLZ1 C1DTLZ3 --m-values 3 5
        """
    )
    
    # Run modes
    parser.add_argument('--all', action='store_true', help='Run all 6 models sequentially')
    parser.add_argument('--algo', type=str, choices=[m['algo'] for m in MODEL_ORDER],
                        help='Run specific algorithm only')
    parser.add_argument('--models', type=str, help='Run specific models by index (0-5), e.g., "0,1,2"')
    
    # Configuration
    parser.add_argument('--workers', type=int, default=16,
                        help='Workers for parallel runs within each model (default: 16)')
    parser.add_argument('--runs', type=int, default=51, help='Runs per config (default: 51)')
    parser.add_argument('--problems', nargs='+', default=ALL_PROBLEMS,
                        help=f'Problems to run (default: all {len(ALL_PROBLEMS)})')
    parser.add_argument('--m-values', nargs='+', type=int, default=ALL_M_VALUES,
                        help=f'M-values to run (default: all {ALL_M_VALUES})')
    parser.add_argument('--output', type=str, default='Results_Sequential',
                        help='Output directory')
    parser.add_argument('--no-resume', action='store_true',
                        help='Do not resume - overwrite existing results')
    
    # Utility
    parser.add_argument('--list', action='store_true', help='List model order and exit')
    parser.add_argument('--status', action='store_true',
                        help='Check status of existing runs and exit')
    
    args = parser.parse_args()
    
    # List mode
    if args.list:
        print("\nModel execution order (lightest → heaviest):\n")
        print(f"{'#':<3} {'Algorithm':<12} {'Complexity':<10} {'Est. Time':<10}")
        print("-" * 40)
        for i, model in enumerate(MODEL_ORDER):
            print(f"{i:<3} {model['algo']:<12} {model['complexity']:<10} ~{model['est_time_hours']}h")
        print()
        return 0
    
    # Status mode
    if args.status:
        print(f"\nChecking status in: {args.output}")
        print("-" * 40)
        for model in MODEL_ORDER:
            done = check_completed(args.output, model['algo'])
            status = "✓ COMPLETE" if done else "○ PENDING"
            print(f"{model['algo']:<12} {status}")
        print()
        return 0
    
    # Validate arguments
    if not (args.all or args.algo or args.models):
        print("Error: Specify --all, --algo, or --models")
        print("Use --list to see model order")
        print("Use --help for more options")
        return 1
    
    # Ensure output directory exists
    Path(args.output).mkdir(parents=True, exist_ok=True)
    
    resume = not args.no_resume
    
    # Run based on mode
    if args.all:
        print("\n🚀 Running ALL 6 models sequentially")
        print(f"Output directory: {args.output}")
        print(f"Workers per model: {args.workers}")
        print(f"Resume mode: {resume}")
        
        completed, failed, results = run_all_sequential(
            problems=args.problems,
            m_values=args.m_values,
            n_runs=args.runs,
            workers=args.workers,
            output_dir=args.output,
            resume=resume,
        )
        
        return 0 if len(failed) == 0 else 1
        
    elif args.algo:
        print(f"\n🚀 Running single model: {args.algo}")
        
        success, elapsed, results = run_single_model(
            algo_name=args.algo,
            problems=args.problems,
            m_values=args.m_values,
            n_runs=args.runs,
            workers=args.workers,
            output_dir=args.output,
            resume=resume,
        )
        
        return 0 if success else 1
        
    elif args.models:
        # Parse indices like "0,1,2" or "0-2"
        indices = []
        for part in args.models.split(','):
            if '-' in part:
                start, end = map(int, part.split('-'))
                indices.extend(range(start, end + 1))
            else:
                indices.append(int(part))
        
        # Validate
        indices = [i for i in indices if 0 <= i < len(MODEL_ORDER)]
        if not indices:
            print("Error: No valid model indices")
            return 1
        
        selected = [MODEL_ORDER[i] for i in indices]
        print(f"\n🚀 Running {len(selected)} selected models: {[m['algo'] for m in selected]}")
        
        # Run with custom model order
        completed, failed, results = run_all_sequential(
            problems=args.problems,
            m_values=args.m_values,
            n_runs=args.runs,
            workers=args.workers,
            output_dir=args.output,
            resume=resume,
            model_order=selected,
        )
        return 0 if len(failed) == 0 else 1


if __name__ == '__main__':
    sys.exit(main())
