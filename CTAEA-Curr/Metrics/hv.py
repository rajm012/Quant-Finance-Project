

"""
Hypervolume (HV) indicator.

HV(P) = volume of objective space dominated by P and bounded by reference point zr.

Uses the WFG algorithm for exact computation (supports many objectives).
For efficiency, a Monte Carlo approximation is also provided.
"""


import numpy as np
# from itertools import product as iproduct


# -----------------------------------------------------------------------
# Exact HV using the recursive WFG algorithm (Fonseca et al.)
# -----------------------------------------------------------------------


def _dominates(p, q):
    """True if p dominates q (minimization)."""
    return np.all(p <= q) and np.any(p < q)



def _exclusive_hv(P, ref, i):
    """Exclusive hypervolume contribution of point P[i] wrt the rest."""
    dominated = [P[j] for j in range(len(P)) if j != i and np.all(P[j] <= P[i])]
    if not dominated:
        return _hv_recursive(np.array([P[i]]), ref)
    
    else:
        return _hv_recursive(np.array([P[i]]), ref) - _hv_recursive(
            np.array(dominated), ref
        )
        


def _limit_set(P, ref_point):
    """Project points onto the boundary of the reference point."""
    return np.minimum(P, ref_point)



def _hv_2d(P, ref):
    """Exact 2D hypervolume."""
    P_sorted = P[np.argsort(P[:, 0])]
    hv = 0.0
    prev_y = ref[1]
    for p in P_sorted:
        if p[1] < prev_y:
            hv += (ref[0] - p[0]) * (prev_y - p[1])
            prev_y = p[1]
    return hv



def _hv_recursive(P, ref):
    """
    Recursive WFG algorithm for exact hypervolume computation.
    Works for any number of objectives.
    """
    n, m = P.shape

    if n == 0:
        return 0.0
    
    if m == 1:
        return ref[0] - np.min(P[:, 0])
    
    if m == 2:
        return _hv_2d(P, ref)

    # Sort by last objective
    P = P[np.argsort(P[:, -1])]
    hv = 0.0
    for i in range(n):
        # Slice: consider only points that dominate P[i] in the last objective
        p_i = P[i]
        if i == 0:
            hv_slice = ref[-1] - p_i[-1]
        else:
            hv_slice = P[i - 1, -1] - p_i[-1]

        if hv_slice <= 0:
            continue

        # Limit set: take points P[0..i] and limit to ref
        limited = np.minimum(P[: i + 1, :-1], ref[:-1])
        
        # Non-dominated front of limited set
        nd = _non_dominated_front(limited)
        hv += hv_slice * _hv_recursive(nd, ref[:-1])
    return hv



def _non_dominated_front(P):
    """Return non-dominated points in P (minimization)."""
    n = len(P)
    dominated = np.zeros(n, dtype=bool)
    for i in range(n):
        for j in range(n):
            if i != j and not dominated[j]:
                if np.all(P[j] <= P[i]) and np.any(P[j] < P[i]):
                    dominated[i] = True
                    break
                
    return P[~dominated]



def hypervolume(P, ref_point):
    """
    Compute exact Hypervolume indicator.
    Parameters
    ----------
    P : np.ndarray, shape (N, m)
        Approximation set.
    ref_point : np.ndarray, shape (m,)
        Reference point (should be dominated by all points in P).

    Returns
    -------
    float
        Hypervolume value. Larger is better.
    """
    
    P = np.array(P)
    ref_point = np.array(ref_point)
    if len(P) == 0:
        return 0.0

    # Remove solutions dominated by or equal to ref_point
    valid = np.all(P < ref_point, axis=1)
    P = P[valid]
    if len(P) == 0:
        return 0.0

    # Remove dominated solutions
    P = _non_dominated_front(P)
    return _hv_recursive(P, ref_point)



def hypervolume_monte_carlo(P, ref_point, n_samples=100000, seed=42):
    """
    Monte Carlo approximation of Hypervolume.
    Useful for high-dimensional cases where exact computation is slow.

    Parameters
    ----------
    P : np.ndarray, shape (N, m)
    ref_point : np.ndarray, shape (m,)
    n_samples : int
    seed : int

    Returns
    -------
    float
        Approximate HV value.
    """
    
    P = np.array(P)
    ref_point = np.array(ref_point)
    if len(P) == 0:
        return 0.0

    # Keep only points that can contribute inside the reference box.
    valid = np.all(P < ref_point, axis=1)
    P = P[valid]
    if len(P) == 0:
        return 0.0

    # Remove dominated points to reduce unnecessary checks.
    P = _non_dominated_front(P)
    if len(P) == 0:
        return 0.0

    rng = np.random.default_rng(seed)
    m = P.shape[1]
    lb = np.min(P, axis=0)
    ub = ref_point
    samples = rng.uniform(lb, ub, size=(n_samples, m))

    # Check if any solution dominates each sample
    dominated = np.zeros(n_samples, dtype=bool)
    
    for p in P:
        # In minimization, sample s is dominated by p iff p <= s.
        dominated |= np.all(p <= samples, axis=1)

    volume = np.prod(ub - lb)
    return volume * np.mean(dominated)


