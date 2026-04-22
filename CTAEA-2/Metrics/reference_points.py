"""
Reference point generation for IGD and HV metric calculation.
Provides approximate Pareto front samples for each problem type.
"""

import numpy as np


# ---------------------------------------------------------------------------
# Generic simplex samplers
# ---------------------------------------------------------------------------

def sample_simplex(m: int, n_points: int) -> np.ndarray:
    """
    Uniformly sample points on the unit simplex (sum = 1, all >= 0).
    Used for DTLZ1-type Pareto fronts.
    """
    points = []
    while len(points) < n_points:
        x = np.random.exponential(1.0, m)
        x /= x.sum()
        points.append(x)
    return np.array(points)


def sample_sphere_front(m: int, n_points: int, radius: float = 1.0) -> np.ndarray:
    """
    Uniformly sample points on the positive orthant of a sphere of given radius.
    Used for DTLZ2/DTLZ3/DTLZ4-type Pareto fronts.
    """
    points = []
    while len(points) < n_points:
        x = np.abs(np.random.randn(m))
        x = x / np.linalg.norm(x) * radius
        points.append(x)
    return np.array(points)


def sample_grid_simplex(m: int, H: int) -> np.ndarray:
    """
    Generate a deterministic grid on the simplex using Das & Dennis method.
    """
    from utils import generate_weight_vectors
    W = generate_weight_vectors(m, H)
    # Scale to form PF points for DTLZ1 (sum = 0.5)
    return W * 0.5


def sample_grid_sphere(m: int, H: int, radius: float = 1.0) -> np.ndarray:
    """
    Generate deterministic grid on the sphere front.
    """
    from utils import generate_weight_vectors
    W = generate_weight_vectors(m, H)
    # Normalize to be on sphere
    norms = np.linalg.norm(W, axis=1, keepdims=True)
    norms[norms < 1e-10] = 1.0
    return W / norms * radius


# ---------------------------------------------------------------------------
# Problem-specific reference point generators
# ---------------------------------------------------------------------------

def get_reference_points(problem_name: str, m: int,
                          n_points: int = 1000) -> np.ndarray:
    """
    Get reference points P* for IGD calculation.
    Returns array of shape (n_points, m).
    """
    # Use higher H for deterministic grids
    H_map = {3: 100, 5: 40, 8: 15, 10: 10, 15: 8}
    H = H_map.get(m, 10)

    if problem_name in ('C1-DTLZ1', 'C3-DTLZ1',
                        'DC1-DTLZ1', 'DC2-DTLZ1', 'DC3-DTLZ1'):
        # PF: sum(f) = 0.5
        P = sample_grid_simplex(m, H)
        if len(P) < n_points:
            P2 = sample_simplex(m, n_points - len(P)) * 0.5
            P = np.vstack([P, P2])
        return P[:n_points]

    elif problem_name in ('C1-DTLZ3', 'C2-DTLZ2',
                          'DC1-DTLZ3', 'DC2-DTLZ3', 'DC3-DTLZ3'):
        # PF: unit sphere in positive orthant
        P = sample_grid_sphere(m, H, radius=1.0)
        if len(P) < n_points:
            P2 = sample_sphere_front(m, n_points - len(P), radius=1.0)
            P = np.vstack([P, P2])
        return P[:n_points]

    elif problem_name == 'C3-DTLZ4':
        # PF: unit sphere (C3-DTLZ4 shares the same PF as DTLZ4)
        P = sample_grid_sphere(m, H, radius=1.0)
        if len(P) < n_points:
            P2 = sample_sphere_front(m, n_points - len(P), radius=1.0)
            P = np.vstack([P, P2])
        return P[:n_points]

    else:
        return sample_sphere_front(m, n_points, radius=1.0)


# ---------------------------------------------------------------------------
# C3-DTLZ1 PF sampler (PF formed by constraint surfaces)
# ---------------------------------------------------------------------------

def sample_c3dtlz1_pf(m: int, n_points: int) -> np.ndarray:
    """
    C3-DTLZ1: PF is the set of points satisfying
        f_j + sum_{i!=j} f_i/0.5 = 1  for at least one j
    and f_i >= 0, sum f_i = 0.5
    This is a subset of the DTLZ1 PF.
    """
    pts = sample_simplex(m, n_points * 10) * 0.5
    valid = []
    for pt in pts:
        for j in range(m):
            cj = pt[j] + sum(pt[i] / 0.5 for i in range(m) if i != j) - 1.0
            if abs(cj) < 0.05:
                valid.append(pt)
                break
    if len(valid) < n_points:
        # Fall back to full simplex
        return sample_simplex(m, n_points) * 0.5
    return np.array(valid[:n_points])


# ---------------------------------------------------------------------------
# Nadir / worst reference point for HV
# ---------------------------------------------------------------------------

def get_reference_point_hv(problem_name: str, m: int) -> np.ndarray:
    """
    Reference point z_r for HV calculation.
    Paper uses (1.1, ..., 1.1)^T except C3-DTLZ4 where z_r = (2.1, ..., 2.1)^T
    """
    if problem_name == 'C3-DTLZ4':
        return np.ones(m) * 2.1
    # For DTLZ1-type problems the PF is at much lower values, so use a larger ref
    if problem_name in ('C1-DTLZ1', 'C3-DTLZ1', 'DC1-DTLZ1',
                        'DC2-DTLZ1', 'DC3-DTLZ1'):
        return np.ones(m) * 1.1
    return np.ones(m) * 1.1
