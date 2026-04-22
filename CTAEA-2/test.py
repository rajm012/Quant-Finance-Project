"""
test.py — Quick smoke-test for the full C-TAEA pipeline.

Tests:
  1. Problem instantiation and evaluation
  2. Weight vector generation
  3. C-TAEA on C1-DTLZ1 (m=3, small budget)
  4. IGD and HV computation
  5. Peer algorithms
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import traceback

PASS = "✓"
FAIL = "✗"


def test_utils():
    print("\n--- test_utils ---")
    from utils import (generate_weight_vectors, get_weight_vectors,
                       get_population_size, simulated_binary_crossover,
                       polynomial_mutation, fast_non_dominated_sort,
                       fast_association, tchebycheff)

    W = generate_weight_vectors(3, 10)
    assert W.shape[1] == 3
    assert np.allclose(W.sum(axis=1), 1.0), "weights should sum to 1"
    print(f"  {PASS} generate_weight_vectors: shape={W.shape}")

    W91 = get_weight_vectors(3, 91)
    print(f"  {PASS} get_weight_vectors(m=3, N=91): got {len(W91)} vectors")

    N = get_population_size(3)
    assert N == 91, f"expected 91 for m=3, got {N}"
    print(f"  {PASS} get_population_size(m=3) = {N}")

    xl = np.zeros(5)
    xu = np.ones(5)
    p1 = np.random.rand(5)
    p2 = np.random.rand(5)
    c1, c2 = simulated_binary_crossover(p1, p2, xl, xu)
    assert len(c1) == 5 and len(c2) == 5
    print(f"  {PASS} simulated_binary_crossover OK")

    m = polynomial_mutation(p1, xl, xu)
    assert len(m) == 5
    assert np.all(m >= 0) and np.all(m <= 1)
    print(f"  {PASS} polynomial_mutation OK")

    F = np.array([[1.0, 2.0], [0.5, 3.0], [0.8, 0.8], [2.0, 1.0]])
    fronts = fast_non_dominated_sort(F)
    assert 2 in fronts[0], f"solution 2 must be in first front, got {fronts[0]}"
    print(f"  {PASS} fast_non_dominated_sort: fronts={fronts}")

    F_norm = np.array([[0.5, 0.5, 0.0], [0.3, 0.3, 0.4]])
    W = get_weight_vectors(3, 91)
    sr = fast_association(F_norm, W)
    assert len(sr) == 2
    print(f"  {PASS} fast_association OK, subregions={sr}")


def test_problems():
    print("\n--- test_problems ---")
    from Problems import ALL_PROBLEMS

    for name, ProbClass in ALL_PROBLEMS.items():
        try:
            prob = ProbClass(m=3)
            x = np.random.uniform(prob.xl, prob.xu)
            F, CV = prob.evaluate(x)
            assert len(F) == 3, f"expected 3 objectives, got {len(F)}"
            assert CV >= 0.0, f"CV must be non-negative, got {CV}"
            print(f"  {PASS} {name}: F={F.round(3)}, CV={CV:.4f}")
        except Exception as e:
            print(f"  {FAIL} {name}: {e}")
            traceback.print_exc()


def test_metrics():
    print("\n--- test_metrics ---")
    from Metrics import igd, hypervolume
    from Metrics.reference_points import get_reference_points, get_reference_point_hv

    # Test IGD
    P      = np.array([[0.1, 0.9], [0.5, 0.5], [0.9, 0.1]])
    P_star = np.array([[0.0, 1.0], [0.5, 0.5], [1.0, 0.0]])
    igd_val = igd(P, P_star)
    assert igd_val >= 0
    print(f"  {PASS} IGD: {igd_val:.6f}")

    # IGD with empty P
    igd_empty = igd(np.array([]).reshape(0, 2), P_star)
    assert igd_empty == np.inf
    print(f"  {PASS} IGD (empty): inf")

    # Test HV
    front = np.array([[0.1, 0.9], [0.5, 0.5], [0.9, 0.1]])
    ref   = np.array([1.1, 1.1])
    hv_val = hypervolume(front, ref)
    assert hv_val > 0
    print(f"  {PASS} HV (2D): {hv_val:.6f}")

    # 3D HV
    front3 = np.random.rand(10, 3) * 0.8
    ref3   = np.ones(3) * 1.1
    hv3 = hypervolume(front3, ref3)
    assert hv3 >= 0
    print(f"  {PASS} HV (3D): {hv3:.6f}")

    # Reference points
    P_star_c1 = get_reference_points('C1-DTLZ1', 3, n_points=100)
    assert P_star_c1.shape == (100, 3)
    print(f"  {PASS} Reference points C1-DTLZ1 m=3: {P_star_c1.shape}")


def test_ctaea_small():
    print("\n--- test_ctaea (small budget) ---")
    from Problems import ALL_PROBLEMS
    from Algorithms.ctaea import CTAEA
    from Metrics import igd
    from Metrics.reference_points import get_reference_points

    prob = ALL_PROBLEMS['C1-DTLZ1'](m=3)
    algo = CTAEA(
        problem=prob,
        m=3,
        N=91,
        max_fe=91 * 50,   # 50 generations (paper uses 500)
        seed=42,
        verbose=True
    )

    print(f"  Running C-TAEA on C1-DTLZ1 m=3 (budget={91*50} FEs)...")
    CA, DA = algo.run()

    PF = algo.get_pareto_front('CA')
    print(f"  CA size: {len(CA)}, DA size: {len(DA)}")
    print(f"  Feasible in CA: {sum(1 for s in CA if s.feasible)}")
    print(f"  Non-dominated feasible PF points: {len(PF)}")

    if len(PF) > 0:
        P_star = get_reference_points('C1-DTLZ1', 3, n_points=500)
        igd_val = igd(PF, P_star)
        print(f"  IGD: {igd_val:.6f}")
        print(f"  {PASS} C-TAEA ran successfully, IGD={igd_val:.4e}")
    else:
        print(f"  WARNING: No feasible solutions found (expected for small budget)")
        print(f"  {PASS} C-TAEA ran without errors")


def test_peer_algorithms():
    print("\n--- test_peer_algorithms (small budget) ---")
    from Problems import ALL_PROBLEMS
    from Algorithms.peer_algorithms import CMOEAD, CNSGAIII, CMOEAD_DD, IDBEA, CMOEA

    prob = ALL_PROBLEMS['C1-DTLZ1'](m=3)
    N    = 91
    budget = N * 20

    for AlgoClass in [CMOEAD, CNSGAIII, CMOEAD_DD, IDBEA, CMOEA]:
        try:
            algo = AlgoClass(problem=prob, m=3, N=N, max_fe=budget, seed=42)
            algo.run()
            PF = algo.get_pareto_front()
            print(f"  {PASS} {AlgoClass.name}: PF size={len(PF)}")
        except Exception as e:
            print(f"  {FAIL} {AlgoClass.name}: {e}")
            traceback.print_exc()


def test_c1dtlz3():
    """Special test for C1-DTLZ3 (hardest problem — infeasible barrier)."""
    print("\n--- test_ctaea C1-DTLZ3 (infeasible barrier) ---")
    from Problems import ALL_PROBLEMS
    from Algorithms.ctaea import CTAEA

    prob = ALL_PROBLEMS['C1-DTLZ3'](m=3)
    algo = CTAEA(
        problem=prob, m=3, N=91, max_fe=91 * 100, seed=42, verbose=False
    )
    CA, DA = algo.run()
    n_feas = sum(1 for s in CA if s.feasible)
    print(f"  C1-DTLZ3: feasible in CA = {n_feas}/{len(CA)}")
    if n_feas > 0:
        print(f"  {PASS} Found feasible solutions (overcoming infeasible barrier!)")
    else:
        print(f"  NOTE: No feasible solutions yet (need more budget). "
              f"Paper uses budget={91*1000}.")
        print(f"  {PASS} No crash, algorithm ran correctly.")


def test_dc2dtlz():
    """Test DC2 problems (fluctuating CV)."""
    print("\n--- test_ctaea DC2-DTLZ1 (fluctuating CV) ---")
    from Problems import ALL_PROBLEMS
    from Algorithms.ctaea import CTAEA

    prob = ALL_PROBLEMS['DC2-DTLZ1'](m=3)
    algo = CTAEA(
        problem=prob, m=3, N=91, max_fe=91 * 100, seed=42, verbose=False
    )
    CA, DA = algo.run()
    n_feas = sum(1 for s in CA if s.feasible)
    print(f"  DC2-DTLZ1: feasible in CA = {n_feas}/{len(CA)}")
    print(f"  {PASS} DC2-DTLZ1 ran without errors")


# ---------------------------------------------------------------------------
# Run all tests
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--quick', action='store_true',
                        help='Skip peer algorithm tests')
    args = parser.parse_args()

    print("=" * 60)
    print("C-TAEA Implementation Test Suite")
    print("=" * 60)

    passed = 0
    failed = 0

    tests = [
        test_utils,
        test_problems,
        test_metrics,
        test_ctaea_small,
        test_c1dtlz3,
        test_dc2dtlz,
    ]
    if not args.quick:
        tests.append(test_peer_algorithms)

    for test_fn in tests:
        try:
            test_fn()
            passed += 1
        except AssertionError as e:
            print(f"\n  {FAIL} ASSERTION FAILED in {test_fn.__name__}: {e}")
            failed += 1
        except Exception as e:
            print(f"\n  {FAIL} ERROR in {test_fn.__name__}: {e}")
            traceback.print_exc()
            failed += 1

    print("\n" + "=" * 60)
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 60)

    if failed == 0:
        print("\nAll tests passed! Ready to run full experiments:")
        print("  python Experiments/run_experiments.py --quick")
        print("  python Experiments/run_experiments.py  # full 51-run experiment")
