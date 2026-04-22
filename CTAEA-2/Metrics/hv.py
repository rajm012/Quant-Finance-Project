"""
Hypervolume (HV) metric.
HV(P) = Lebesgue measure of space dominated by P and bounded by z_r.
        (Eq. 15 in paper)

Uses the WFG algorithm for exact computation (While et al. 2012).
For high-dimensional cases (m > 5), we use Monte Carlo estimation.
"""

import numpy as np


# ---------------------------------------------------------------------------
# WFG-based exact hypervolume (for m <= 6 typically)
# ---------------------------------------------------------------------------

def dominated_hypervolume(front: np.ndarray, reference_point: np.ndarray) -> float:
    """
    Compute exact hypervolume using recursive WFG algorithm.
    front            : ndarray (n, m) — non-dominated solutions (minimization)
    reference_point  : ndarray (m,)  — worst point dominated by all PF solutions
    Returns hypervolume value (higher is better).
    """
    front = np.array(front)
    reference_point = np.array(reference_point)

    # Remove solutions not dominated by reference_point (i.e., any f_i >= r_i)
    dominated = np.all(front < reference_point, axis=1)
    front = front[dominated]

    if len(front) == 0:
        return 0.0

    m = front.shape[1]
    if m == 1:
        return float(reference_point[0] - np.min(front[:, 0]))

    return _wfg(front, reference_point)


def _wfg(front: np.ndarray, reference: np.ndarray) -> float:
    """Recursive WFG hypervolume computation."""
    if len(front) == 0:
        return 0.0

    m = front.shape[1]

    # Base case: 1D
    if m == 1:
        return float(reference[0] - np.min(front[:, 0]))

    # Sort by last objective (ascending)
    sorted_idx = np.argsort(front[:, -1])
    front = front[sorted_idx]

    hv = 0.0
    prev_last = reference[-1]

    for i in range(len(front) - 1, -1, -1):
        # Slice: contribution of this point in last dimension
        slice_height = prev_last - front[i, -1]
        if slice_height > 0:
            # Exclusive hypervolume in lower dimensions
            sub_front = _limit_set(front[:i + 1], front[i])
            hv += slice_height * _wfg(sub_front, reference[:-1])
        prev_last = front[i, -1]

    return hv


def _limit_set(front: np.ndarray, point: np.ndarray) -> np.ndarray:
    """Limit the front by taking component-wise max with point (all dims except last)."""
    limited = np.maximum(front[:, :-1], point[:-1])
    return limited


# ---------------------------------------------------------------------------
# Monte Carlo HV estimation (for large m)
# ---------------------------------------------------------------------------

def hv_monte_carlo(front: np.ndarray, reference_point: np.ndarray,
                   n_samples: int = 100000) -> float:
    """
    Monte Carlo HV estimation.
    Samples uniformly in [ideal, reference] box and counts dominated points.
    """
    front = np.array(front)
    dominated = np.all(front < reference_point, axis=1)
    front = front[dominated]
    if len(front) == 0:
        return 0.0

    m = front.shape[1]
    ideal = np.min(front, axis=0)
    box_vol = np.prod(reference_point - ideal)
    if box_vol <= 0:
        return 0.0

    # Sample random points in the box
    samples = np.random.uniform(ideal, reference_point, (n_samples, m))

    # Count how many are dominated by at least one solution in front
    dominated_count = 0
    for s in samples:
        if np.any(np.all(front <= s, axis=1)):
            dominated_count += 1

    return box_vol * (dominated_count / n_samples)


# ---------------------------------------------------------------------------
# Main HV function
# ---------------------------------------------------------------------------

def hypervolume(front: np.ndarray, reference_point: np.ndarray,
                use_mc_threshold: int = 6,
                n_mc_samples: int = 50000) -> float:
    """
    Compute hypervolume.
    Uses exact WFG for m <= use_mc_threshold, Monte Carlo otherwise.

    Parameters
    ----------
    front           : ndarray (n, m)
    reference_point : ndarray (m,)
    use_mc_threshold: int — use exact method if m <= this value
    n_mc_samples    : int — number of MC samples for large m
    """
    front = np.array(front, dtype=float)
    reference_point = np.array(reference_point, dtype=float)

    if len(front) == 0:
        return 0.0

    m = front.shape[1]

    if m <= use_mc_threshold:
        return dominated_hypervolume(front, reference_point)
    else:
        return hv_monte_carlo(front, reference_point, n_mc_samples)
