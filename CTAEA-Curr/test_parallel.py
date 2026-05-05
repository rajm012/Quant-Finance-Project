#!/usr/bin/env python3
"""
Quick test script to verify parallel runner works correctly.
Run this before the full 51-run replication.
"""

import sys
import multiprocessing as mp
import time

# CRITICAL: Set spawn method at the very beginning
if __name__ == '__main__':
    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        pass

def test_basic():
    """Test with a very small configuration first."""
    print("="*70)
    print("TEST 1: Basic Parallel Execution (2 configs, 2 runs each)")
    print("="*70)
    
    from Experiments.parallel_runner import run_experiment_parallel
    
    start = time.time()
    
    results = run_experiment_parallel(
        problem_names=['C1DTLZ1'],
        m_values=[3],
        algo_names=['C-TAEA'],
        n_runs=2,
        n_workers=2,
        output_dir='Results_Test_Basic',
        verbose=True,
    )
    
    elapsed = time.time() - start
    print(f"\nTest completed in {elapsed:.1f} seconds")
    
    if len(results) > 0:
        print("✓ Basic test PASSED")
        return True
    else:
        print("✗ Basic test FAILED")
        return False

def test_small_batch():
    """Test with a slightly larger batch."""
    print("\n" + "="*70)
    print("TEST 2: Small Batch (2 problems × 2 m-values × 2 algos × 3 runs = 24 tasks)")
    print("="*70)
    
    from Experiments.parallel_runner import run_experiment_parallel
    
    start = time.time()
    
    results = run_experiment_parallel(
        problem_names=['C1DTLZ1', 'C1DTLZ3'],
        m_values=[3, 5],
        algo_names=['C-TAEA', 'C-NSGA-III'],
        n_runs=3,
        n_workers=4,
        output_dir='Results_Test_Small',
        verbose=True,
    )
    
    elapsed = time.time() - start
    print(f"\nTest completed in {elapsed:.1f} seconds")
    
    # Should have 2×2×2 = 8 configs in results
    config_count = sum(len(v) for p in results.values() for v in p.values())
    print(f"Configs returned: {config_count}")
    
    if config_count == 8:
        print("✓ Small batch test PASSED")
        return True
    else:
        print("✗ Small batch test FAILED (expected 8 configs)")
        return False

def test_worker_scaling():
    """Test different worker counts to find optimal."""
    print("\n" + "="*70)
    print("TEST 3: Worker Scaling Test")
    print("="*70)
    
    from Experiments.parallel_runner import run_experiment_parallel
    import multiprocessing
    
    cpu_count = multiprocessing.cpu_count()
    print(f"Detected CPUs: {cpu_count}")
    
    # Test with 1, 2, 4 workers
    worker_counts = [1, 2, 4]
    results = {}
    
    for n_workers in worker_counts:
        if n_workers > cpu_count:
            continue
            
        print(f"\nTesting with {n_workers} workers...")
        start = time.time()
        
        _ = run_experiment_parallel(
            problem_names=['C1DTLZ1'],
            m_values=[3],
            algo_names=['C-TAEA'],
            n_runs=4,
            n_workers=n_workers,
            output_dir=f'Results_Test_{n_workers}',
            verbose=False,
        )
        
        elapsed = time.time() - start
        results[n_workers] = elapsed
        print(f"  {n_workers} workers: {elapsed:.2f}s")
    
    if len(results) > 1:
        speedup = results[1] / results[max(results.keys())]
        print(f"\nSpeedup: {speedup:.1f}x with {max(results.keys())} workers")
    
    return True

def main():
    import time
    import argparse
    
    parser = argparse.ArgumentParser(description='Test parallel runner')
    parser.add_argument('--scaling', action='store_true', help='Run scaling test')
    parser.add_argument('--full', action='store_true', help='Run all tests')
    parser.add_argument('--quick', action='store_true', help='Quick test only')
    
    args = parser.parse_args()
    
    print("\n" + "#"*70)
    print("# C-TAEA Parallel Runner Test Suite")
    print("#"*70)
    print("# Using multiprocessing 'spawn' method to avoid deadlocks")
    print("#"*70 + "\n")
    
    all_passed = True
    
    if args.quick:
        # Just run the basic test
        all_passed = test_basic()
    elif args.scaling:
        all_passed = test_worker_scaling()
    elif args.full:
        all_passed = test_basic() and test_small_batch() and test_worker_scaling()
    else:
        # Default: basic + small batch
        all_passed = test_basic() and test_small_batch()
    
    print("\n" + "="*70)
    if all_passed:
        print("ALL TESTS PASSED ✓")
        print("\nYou can now run the full replication:")
        print("  python -m Experiments.parallel_runner --full --workers 48")
    else:
        print("SOME TESTS FAILED ✗")
        print("\nTroubleshooting:")
        print("  1. Make sure you're in the CTAEA directory")
        print("  2. Check that all dependencies are installed: pip install numpy")
        print("  3. Try running with fewer workers: --workers 2")
        print("  4. Check the error messages above")
    print("="*70)
    
    return 0 if all_passed else 1

if __name__ == '__main__':
    sys.exit(main())
