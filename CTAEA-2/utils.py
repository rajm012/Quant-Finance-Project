"""
Utility functions for C-TAEA implementation.
Includes weight vector generation (Das & Dennis / layered approach),
normalization, and other helpers.
"""

import numpy as np
from itertools import combinations_with_replacement


# ---------------------------------------------------------------------------
# Weight vector generation
# ---------------------------------------------------------------------------

def generate_weight_vectors(m: int, H: int) -> np.ndarray:
    """
    Generate uniformly distributed weight vectors on the canonical simplex
    using the Das & Dennis method.

    Parameters
    ----------
    m : int   - number of objectives
    H : int   - number of divisions per axis

    Returns
    -------
    W : ndarray, shape (N, m)  where N = C(H+m-1, m-1)
    """
    def _recursive(m, left, current, result):
        if m == 1:
            result.append(current + [left])
            return
        for i in range(left + 1):
            _recursive(m - 1, left - i, current + [i], result)

    result = []
    _recursive(m, H, [], result)
    W = np.array(result, dtype=float) / H
    return W


def get_H_for_size(m: int, target_N: int) -> int:
    """
    Find H such that the number of weight vectors is >= target_N.
    Uses two-layer approach for large m.
    """
    from math import comb
    for H in range(1, 50):
        N = comb(H + m - 1, m - 1)
        if N >= target_N:
            return H
    return 10


def two_layer_weight_vectors(m: int, target_N: int) -> np.ndarray:
    """
    Two-layer weight vector generation for many-objective problems.
    Layer 1: boundary weights (H1 divisions)
    Layer 2: inner weights  (H2 divisions, scaled toward centroid)
    Follows the approach in Li et al. 2015 (C-MOEA/DD reference).
    """
    from math import comb

    # Find best single-layer H
    W1 = None
    for H in range(1, 20):
        W = generate_weight_vectors(m, H)
        if len(W) >= target_N:
            W1 = W
            break

    if W1 is None or len(W1) >= target_N:
        W1 = generate_weight_vectors(m, 3)

    # Try adding a second, inner layer
    for H2 in range(1, 10):
        W2_raw = generate_weight_vectors(m, H2)
        # Scale inner weights toward centroid
        W2 = W2_raw / 2 + 1 / (2 * m)
        W_combined = np.vstack([W1, W2])
        if len(W_combined) >= target_N:
            return W_combined

    return W1


def get_weight_vectors(m: int, population_size: int) -> np.ndarray:
    """
    Generate weight vectors matching the paper's population sizes.
    Table IV:  m=3->91, m=5->210, m=8->156, m=10->275, m=15->135
    """
    # Single-layer search
    from math import comb
    best_W = None
    best_diff = 1e9
    for H in range(1, 30):
        n = comb(H + m - 1, m - 1)
        diff = abs(n - population_size)
        if diff < best_diff:
            best_diff = diff
            best_W = generate_weight_vectors(m, H)
        if n > population_size * 2:
            break

    # Two-layer if single-layer is not close enough
    for H1 in range(1, 15):
        n1 = comb(H1 + m - 1, m - 1)
        if n1 >= population_size:
            break
        for H2 in range(1, 15):
            n2 = comb(H2 + m - 1, m - 1)
            total = n1 + n2
            diff = abs(total - population_size)
            if diff < best_diff:
                best_diff = diff
                W1 = generate_weight_vectors(m, H1)
                W2_raw = generate_weight_vectors(m, H2)
                W2 = W2_raw / 2 + 1 / (2 * m)
                best_W = np.vstack([W1, W2])

    # Ensure no zero weights (replace with small epsilon)
    best_W = np.clip(best_W, 1e-6, 1.0)
    return best_W


# ---------------------------------------------------------------------------
# Population size lookup (Table IV from paper)
# ---------------------------------------------------------------------------

POPULATION_SIZES = {
    3:  91,
    5:  210,
    8:  156,
    10: 275,
    15: 135,
}


def get_population_size(m: int) -> int:
    return POPULATION_SIZES.get(m, 100)


# ---------------------------------------------------------------------------
# Function evaluation budget (Table V)
# ---------------------------------------------------------------------------

FE_BUDGETS = {
    'C1-DTLZ1':  {3: 500,  5: 600,  8: 800,  10: 1000, 15: 1500},
    'C1-DTLZ3':  {3: 1000, 5: 1500, 8: 2500, 10: 3500, 15: 5000},
    'C2-DTLZ2':  {3: 250,  5: 350,  8: 500,  10: 750,  15: 1000},
    'C3-DTLZ1':  {3: 750,  5: 1250, 8: 2000, 10: 3000, 15: 4000},
    'C3-DTLZ4':  {3: 750,  5: 1250, 8: 2000, 10: 3000, 15: 4000},
    'DC1-DTLZ1': {3: 500,  5: 600,  8: 800,  10: 1000, 15: 1500},
    'DC1-DTLZ3': {3: 1000, 5: 1500, 8: 2500, 10: 3500, 15: 5000},
    'DC2-DTLZ1': {3: 500,  5: 600,  8: 800,  10: 1000, 15: 1500},
    'DC2-DTLZ3': {3: 1000, 5: 1500, 8: 2500, 10: 3500, 15: 5000},
    'DC3-DTLZ1': {3: 750,  5: 1250, 8: 2000, 10: 3000, 15: 4000},
    'DC3-DTLZ3': {3: 750,  5: 1250, 8: 2000, 10: 3000, 15: 4000},
}


def get_fe_budget(problem_name: str, m: int) -> int:
    budgets = FE_BUDGETS.get(problem_name, {})
    N = get_population_size(m)
    multiplier = budgets.get(m, 500)
    return multiplier * N


# ---------------------------------------------------------------------------
# Normalization helpers
# ---------------------------------------------------------------------------

def normalize_objectives(F: np.ndarray, z_ideal: np.ndarray,
                          z_nadir: np.ndarray) -> np.ndarray:
    """Normalize objective vectors using ideal and nadir points."""
    denom = z_nadir - z_ideal
    denom[denom < 1e-10] = 1e-10
    return (F - z_ideal) / denom


# ---------------------------------------------------------------------------
# Angle / subregion association
# ---------------------------------------------------------------------------

def associate_to_subregion(F_norm: np.ndarray, W: np.ndarray) -> np.ndarray:
    """
    Associate each solution to the closest weight vector (smallest angle).
    Returns index array of shape (n_solutions,).
    """
    # Compute angle between each solution and each weight vector
    # angle(F, w) = arccos(F·w / (|F||w|))
    # For association we just use the acute angle (dot product after
    # normalizing both F and w).
    n = F_norm.shape[0]
    indices = np.zeros(n, dtype=int)
    for i in range(n):
        f = F_norm[i]
        f_norm = np.linalg.norm(f)
        if f_norm < 1e-10:
            indices[i] = 0
            continue
        # angle with each weight vector
        angles = np.array([
            angle_between(f, W[j]) for j in range(len(W))
        ])
        indices[i] = np.argmin(angles)
    return indices


def angle_between(f: np.ndarray, w: np.ndarray) -> float:
    """Acute angle between vectors f and w."""
    fn = np.linalg.norm(f)
    wn = np.linalg.norm(w)
    if fn < 1e-10 or wn < 1e-10:
        return np.pi / 2
    cos_val = np.dot(f, w) / (fn * wn)
    cos_val = np.clip(cos_val, -1.0, 1.0)
    return np.arccos(cos_val)


def perpendicular_distance(f: np.ndarray, w: np.ndarray) -> float:
    """
    Perpendicular distance from point f to the line defined by weight vector w.
    d_perp(f, w) = || f - (f·w / |w|^2) w ||
    Used in Algorithm 1 of the paper.
    """
    w_norm_sq = np.dot(w, w)
    if w_norm_sq < 1e-10:
        return np.linalg.norm(f)
    proj = np.dot(f, w) / w_norm_sq * w
    return np.linalg.norm(f - proj)


def fast_association(F_norm: np.ndarray, W: np.ndarray) -> np.ndarray:
    """
    Vectorized association: assign each solution to the weight vector
    with minimum perpendicular distance (as in Algorithm 1).
    Returns index array.
    """
    n_sol = F_norm.shape[0]
    n_w   = W.shape[0]

    # d_perp(f, w) = || f - (f·w/|w|^2)*w ||
    # Shape: (n_sol, n_w)
    W_norm_sq = np.sum(W ** 2, axis=1)  # (n_w,)
    dots = F_norm @ W.T                 # (n_sol, n_w)
    projs = dots / W_norm_sq            # (n_sol, n_w)  scalar projections

    # projected vectors: (n_sol, n_w, m)
    proj_vecs = projs[:, :, np.newaxis] * W[np.newaxis, :, :]
    diff = F_norm[:, np.newaxis, :] - proj_vecs  # (n_sol, n_w, m)
    dists = np.linalg.norm(diff, axis=2)          # (n_sol, n_w)

    return np.argmin(dists, axis=1)


# ---------------------------------------------------------------------------
# Tchebycheff aggregation
# ---------------------------------------------------------------------------

def tchebycheff(f_norm: np.ndarray, w: np.ndarray, z_ideal: np.ndarray) -> float:
    """
    Weighted Tchebycheff aggregation (Eq. 11 in paper).
    g^tch(x | w, z*) = max_j { |f_j(x) - z*_j| / w_j }
    """
    w_safe = np.where(w < 1e-6, 1e-6, w)
    return np.max(np.abs(f_norm - z_ideal) / w_safe)


def tchebycheff_batch(F_norm: np.ndarray, W: np.ndarray,
                      z_ideal: np.ndarray) -> np.ndarray:
    """
    Compute Tchebycheff values for multiple solutions and weight vectors.
    F_norm: (n,)  - normalized objective vector
    W:      (N, m)
    Returns: (N,) array
    """
    W_safe = np.where(W < 1e-6, 1e-6, W)
    diffs = np.abs(F_norm - z_ideal)           # (m,)
    return np.max(diffs / W_safe, axis=1)      # (N,)


# ---------------------------------------------------------------------------
# Non-dominated sorting (fast NSGA-II style)
# ---------------------------------------------------------------------------

def fast_non_dominated_sort(F: np.ndarray) -> list:
    """
    Fast non-dominated sorting — vectorized implementation.
    Returns list of fronts, each front is a list of solution indices.
    """
    n = len(F)
    if n == 0:
        return []

    # Vectorized dominance: dom_matrix[i,j] = True if i dominates j
    # i dominates j iff F[i] <= F[j] everywhere AND F[i] < F[j] somewhere
    # Shape: (n, n, m) -> broadcast
    F_i = F[:, np.newaxis, :]   # (n,1,m)
    F_j = F[np.newaxis, :, :]   # (1,n,m)
    leq  = np.all(F_i <= F_j, axis=2)   # (n,n)
    less = np.any(F_i <  F_j, axis=2)   # (n,n)
    dom  = leq & less                    # i dominates j

    domination_count = dom.sum(axis=0)   # how many solutions dominate each sol
    np.fill_diagonal(dom, False)

    fronts = []
    remaining = np.ones(n, dtype=bool)

    while remaining.any():
        # Current front: solutions with domination_count == 0 among remaining
        front_mask = remaining & (domination_count == 0)
        if not front_mask.any():
            # Numerical issues — remaining all go to last front
            fronts.append(list(np.where(remaining)[0]))
            break

        front = list(np.where(front_mask)[0])
        fronts.append(front)

        # Update counts: remove contributions from this front
        for i in front:
            remaining[i] = False
            dominated_by_i = np.where(dom[i])[0]
            domination_count[dominated_by_i] -= 1

    return fronts


def _dominates(f1: np.ndarray, f2: np.ndarray) -> bool:
    """Return True if f1 Pareto-dominates f2."""
    return np.all(f1 <= f2) and np.any(f1 < f2)


def dominates_batch(f: np.ndarray, F: np.ndarray) -> np.ndarray:
    """Returns bool array: does f dominate each row of F?"""
    return np.all(f <= F, axis=1) & np.any(f < F, axis=1)


# ---------------------------------------------------------------------------
# Reproduction operators
# ---------------------------------------------------------------------------

def simulated_binary_crossover(p1: np.ndarray, p2: np.ndarray,
                                 xl: np.ndarray, xu: np.ndarray,
                                 eta_c: float = 30.0,
                                 prob_c: float = 0.9) -> tuple:
    """
    Simulated Binary Crossover (SBX).
    Returns two offspring.
    """
    n = len(p1)
    c1, c2 = p1.copy(), p2.copy()

    if np.random.random() > prob_c:
        return c1, c2

    for i in range(n):
        if np.random.random() <= 0.5:
            if abs(p1[i] - p2[i]) > 1e-10:
                y1, y2 = min(p1[i], p2[i]), max(p1[i], p2[i])
                rand = np.random.random()
                # beta
                beta = 1.0 + (2.0 * (y1 - xl[i]) / (y2 - y1))
                alpha = 2.0 - beta ** (-(eta_c + 1))
                betaq = _sbx_betaq(rand, alpha, eta_c)
                c1[i] = 0.5 * ((y1 + y2) - betaq * (y2 - y1))

                beta = 1.0 + (2.0 * (xu[i] - y2) / (y2 - y1))
                alpha = 2.0 - beta ** (-(eta_c + 1))
                betaq = _sbx_betaq(rand, alpha, eta_c)
                c2[i] = 0.5 * ((y1 + y2) + betaq * (y2 - y1))

                c1[i] = np.clip(c1[i], xl[i], xu[i])
                c2[i] = np.clip(c2[i], xl[i], xu[i])

                if np.random.random() > 0.5:
                    c1[i], c2[i] = c2[i], c1[i]

    return c1, c2


def _sbx_betaq(rand: float, alpha: float, eta_c: float) -> float:
    if rand <= 1.0 / alpha:
        return (rand * alpha) ** (1.0 / (eta_c + 1))
    else:
        return (1.0 / (2.0 - rand * alpha)) ** (1.0 / (eta_c + 1))


def polynomial_mutation(x: np.ndarray, xl: np.ndarray, xu: np.ndarray,
                         eta_m: float = 20.0,
                         prob_m: float) -> np.ndarray:
    """
    Polynomial mutation.
    prob_m defaults to 1/n (as in paper).
    """
    n = len(x)
    if prob_m is None:
        prob_m = 1.0 / n

    x_mut = x.copy()
    for i in range(n):
        if np.random.random() < prob_m:
            delta1 = (x[i] - xl[i]) / (xu[i] - xl[i] + 1e-10)
            delta2 = (xu[i] - x[i]) / (xu[i] - xl[i] + 1e-10)
            rand = np.random.random()
            mut_pow = 1.0 / (eta_m + 1.0)

            if rand < 0.5:
                xy = 1.0 - delta1
                val = 2.0 * rand + (1.0 - 2.0 * rand) * (xy ** (eta_m + 1))
                deltaq = val ** mut_pow - 1.0
            else:
                xy = 1.0 - delta2
                val = 2.0 * (1.0 - rand) + 2.0 * (rand - 0.5) * (xy ** (eta_m + 1))
                deltaq = 1.0 - val ** mut_pow

            x_mut[i] = x[i] + deltaq * (xu[i] - xl[i])
            x_mut[i] = np.clip(x_mut[i], xl[i], xu[i])

    return x_mut
