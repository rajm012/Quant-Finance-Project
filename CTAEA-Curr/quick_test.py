#!/usr/bin/env python3
"""
Ultra-quick test to verify parallel execution works.
This should complete in under 30 seconds.
"""

import sys
import multiprocessing as mp

# CRITICAL: Set spawn method at the very beginning
if __name__ == '__main__':
    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        pass

    print("Testing C-TAEA parallel execution...")
    print(f"Python: {sys.version}")
    print(f"Multiprocessing start method: {mp.get_start_method()}")
    print()
    
    # Test basic import
    try:
        from Experiments.parallel_runner import run_experiment_parallel
        print("✓ Import successful")
    except Exception as e:
        print(f"✗ Import failed: {e}")
        sys.exit(1)
    
    # Test with minimal configuration (1 problem, 1 m, 1 algo, 2 runs, 2 workers)
    print("\nRunning minimal test (2 runs, 2 workers)...")
    print("-" * 50)
    
    try:
        results = run_experiment_parallel(
            problem_names=['C1DTLZ1'],
            m_values=[3],
            algo_names=['C-TAEA'],
            n_runs=2,
            n_workers=2,
            output_dir='Results_QuickTest',
            verbose=True,
        )
        
        print("-" * 50)
        
        if len(results) > 0:
            print("\n✓ QUICK TEST PASSED")
            print("\nResults preview:")
            for prob, mdict in results.items():
                for m, adict in mdict.items():
                    for algo, data in adict.items():
                        print(f"  {prob} m={m} {algo}: IGD={data.get('igd_median', 'N/A'):.4e}")
            
            print("\nYou can now run full replication:")
            print("  python main.py --full --workers 48")
            sys.exit(0)
        else:
            print("\n✗ Test returned no results")
            sys.exit(1)
            
    except Exception as e:
        import traceback
        print(f"\n✗ Test failed with error: {e}")
        traceback.print_exc()
        sys.exit(1)
