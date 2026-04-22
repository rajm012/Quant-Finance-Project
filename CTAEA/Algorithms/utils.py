

"""
Utility functions for C-TAEA algorithm.
Includes: weight vector generation, normalization, association, dominance checks.
"""


import numpy as np
# from itertools import combinations_with_replacement


# ─────────────────────────────────────────────────────────────────────────────
# Weight vector generation (Das-Dennis simplex lattice)
# ─────────────────────────────────────────────────────────────────────────────

def generate_weight_vectors(m, H):
    """
    Generate uniformly distributed weight vectors on a canonical simplex.
    Parameters
    ----------
    m : int   Number of objectives.
    H : int   Number of divisions.
    
    Returns
    -------
    W : np.ndarray, shape (C(H+m-1, m-1), m)
    """
    def _gen(m, left, current):
        if m == 1:
            yield current + [left]
        else:
            for i in range(left + 1):
                yield from _gen(m - 1, left - i, current + [i])

    W = np.array(list(_gen(m, H, [])), dtype=float)
    return W / H



def get_N_and_H(m):
    """
    Return (N, H) matching Table IV of the paper.
    For m in {3,5,8,10,15}.
    Uses single-layer or two-layer weight vector generation.
    """
    table = {
        3:  (91,  12),
        5:  (210, 6),
        8:  (156, 4),
        10: (275, 4),
        15: (135, 3),
    }
    return table.get(m, (100, 5))


def generate_reference_vectors_table_iv(m):
    """Generate reference vectors matching Table IV sizes.

    For m in {8, 10, 15}, use a two-layer construction to produce exactly
    N reference vectors, consistent with common NSGA-III/C-TAEA setups.
    """
    if m == 3:
        return generate_weight_vectors(3, 12)
    if m == 5:
        return generate_weight_vectors(5, 6)

    # Two-layer settings chosen to match Table IV cardinalities exactly:
    # m=8  : C(3+8-1,8-1)=120 and C(2+8-1,8-1)=36  -> 156
    # m=10 : C(3+10-1,10-1)=220 and C(2+10-1,10-1)=55 -> 275
    # m=15 : C(2+15-1,15-1)=120 and C(1+15-1,15-1)=15 -> 135
    two_layer = {
        8: (3, 2),
        10: (3, 2),
        15: (2, 1),
    }

    if m not in two_layer:
        _, H = get_N_and_H(m)
        return generate_weight_vectors(m, H)

    H1, H2 = two_layer[m]
    W1 = generate_weight_vectors(m, H1)
    W2 = generate_weight_vectors(m, H2)

    # Inner layer shrink toward simplex center (standard two-layer trick).
    center = 1.0 / m
    W2 = 0.5 * W2 + 0.5 * center

    return np.vstack([W1, W2])



def get_population_size(m):
    """Return population size N for given m objectives (from Table IV)."""
    N, _ = get_N_and_H(m)
    return N


# ─────────────────────────────────────────────────────────────────────────────
# Objective normalization
# ─────────────────────────────────────────────────────────────────────────────


def normalize_objectives(F, z_ideal, z_nadir):
    """
    Normalize objective values to [0, 1].

    f_norm = (f - z_ideal) / (z_nadir - z_ideal)

    Parameters
    ----------
    F      : (N, m) objective matrix
    z_ideal: (m,) ideal point
    z_nadir: (m,) nadir point

    Returns
    -------
    F_norm : (N, m)
    """
    denom = z_nadir - z_ideal
    denom = np.where(denom < 1e-10, 1e-10, denom)
    return (F - z_ideal) / denom



# ─────────────────────────────────────────────────────────────────────────────
# Tchebycheff scalarization
# ─────────────────────────────────────────────────────────────────────────────


def tchebycheff(F_norm, W, z_ideal_norm=None):
    """
    Compute Tchebycheff (weighted Chebyshev) aggregation values.
    g_tch(x|w, z*) = max_j { |f_j(x) - z*_j| / w_j }
    Parameters
    ----------
    F_norm     : (N, m) normalized objectives
    W          : (N_w, m) weight vectors  OR (m,) single weight vector
    z_ideal_norm: (m,) ideal point in normalized space (default zeros)

    Returns
    -------
    G : (N, N_w) or (N,) tchebycheff values
    """
    if z_ideal_norm is None:
        z_ideal_norm = np.zeros(F_norm.shape[1])

    diff = np.abs(F_norm - z_ideal_norm)  # (N, m)
    if W.ndim == 1:
        w = np.where(W < 1e-6, 1e-6, W)
        return np.max(diff / w, axis=1)
    
    else:
        W_safe = np.where(W < 1e-6, 1e-6, W)   # (N_w, m)
        # (N, 1, m) / (1, N_w, m) -> (N, N_w, m) -> max over m -> (N, N_w)
        return np.max(diff[:, np.newaxis, :] / W_safe[np.newaxis, :, :], axis=2)



# ─────────────────────────────────────────────────────────────────────────────
# Subregion association
# ─────────────────────────────────────────────────────────────────────────────


def associate_to_subregions(F_norm, W):
    """
    Associate each solution to the subregion (weight vector) with smallest perpendicular distance.
    In practice, the paper uses acute-angle criterion:
      k = argmin_i angle(F(x), w_i)  <==>  argmin_i d_perp(x, w_i)

    d_perp(x, w) = ||x - (w^T x / ||w||^2) * w||

    Parameters
    ----------
    F_norm : (N, m)
    W      : (N_w, m)

    Returns
    -------
    assignments : (N,) int array, index into W for each solution
    """
    
    N = F_norm.shape[0]
    N_w = W.shape[0]

    # Perpendicular distance from each solution to each weight vector line
    # d_perp(x, w) = ||x||^2 - (w^T x)^2 / ||w||^2
    # Equivalent to: ||x - proj(x, w)||
    
    # dot products: (N, N_w)
    # projected lengths squared
    # ||F||^2 for each solution
    # d_perp^2
    
    W_norm_sq = np.sum(W ** 2, axis=1)  # (N_w,)
    dots = F_norm @ W.T  # (N, N_w)
    proj_sq = dots ** 2 / W_norm_sq[np.newaxis, :]  # (N, N_w)
    F_sq = np.sum(F_norm ** 2, axis=1, keepdims=True)  # (N, 1)
    d_perp_sq = np.maximum(F_sq - proj_sq, 0.0)  # (N, N_w)
    assignments = np.argmin(d_perp_sq, axis=1)  # (N,)
    return assignments



# ─────────────────────────────────────────────────────────────────────────────
# Pareto dominance
# ─────────────────────────────────────────────────────────────────────────────


def dominates(f1, f2):
    """True if f1 dominates f2 (both 1D arrays, minimization)."""
    return np.all(f1 <= f2) and np.any(f1 < f2)


def fast_non_dominated_sort(F):
    """
    Fast non-dominated sorting (NSGA-II style).
    Parameters
    ----------
    F : (N, m) objective values

    Returns
    -------
    fronts : list of lists
        fronts[0] = list of indices in first non-dominated front, etc.
    rank   : (N,) rank of each individual (0 = first front)
    """
    N = len(F)
    S = [[] for _ in range(N)]      # dominated set
    n_dom = np.zeros(N, dtype=int)  # domination counter
    rank = np.zeros(N, dtype=int)

    for p in range(N):
        for q in range(N):
            if p == q:
                continue
            if dominates(F[p], F[q]):
                S[p].append(q)
            elif dominates(F[q], F[p]):
                n_dom[p] += 1

    fronts = []
    current_front = []
    for p in range(N):
        if n_dom[p] == 0:
            rank[p] = 0
            current_front.append(p)

    fronts.append(current_front)
    i = 0
    while fronts[i]:
        next_front = []
        for p in fronts[i]:
            for q in S[p]:
                n_dom[q] -= 1
                if n_dom[q] == 0:
                    rank[q] = i + 1
                    next_front.append(q)
        i += 1
        fronts.append(next_front)

    # Remove empty last front
    if not fronts[-1]:
        fronts = fronts[:-1]

    return fronts, rank



def fast_non_dominated_sort_with_cv(F, CV):
    """
    Non-dominated sorting using the bi-objective problem (CV, g_tch).
    Used in the CA update when feasible solutions are insufficient.
    Eq. (12) in the paper.

    Parameters
    ----------
    F  : (N, m) objectives (not used directly; we sort on CV + g_tch)
    CV : (N,) constraint violations

    Returns
    -------
    Same as fast_non_dominated_sort but on the (CV, g_tch) bi-objective space.
    Here we pass a combined 2-col matrix externally.
    """
    return fast_non_dominated_sort(F)


def non_dominated_indices(F):
    """Return boolean mask of non-dominated solutions."""
    N = len(F)
    nd = np.ones(N, dtype=bool)
    for i in range(N):
        for j in range(N):
            if i != j and nd[i]:
                if dominates(F[j], F[i]):
                    nd[i] = False
                    break
    return nd



# ─────────────────────────────────────────────────────────────────────────────
# Simulated Binary Crossover (SBX) and Polynomial Mutation
# ─────────────────────────────────────────────────────────────────────────────


def sbx_crossover(p1, p2, xl, xu, eta_c=30.0, pc=0.9):
    """
    Simulated Binary Crossover.
    Parameters
    ----------
    p1, p2 : (n,) parent decision vectors
    xl, xu : (n,) lower / upper bounds
    eta_c  : distribution index (default 30, as per paper Table III)
    pc     : crossover probability (default 0.9)

    Returns
    -------
    c1, c2 : (n,) offspring
    """
    n = len(p1)
    c1, c2 = p1.copy(), p2.copy()

    if np.random.rand() > pc:
        return c1, c2

    for i in range(n):
        if np.random.rand() <= 0.5:
            if abs(p1[i] - p2[i]) > 1e-14:
                y1, y2 = min(p1[i], p2[i]), max(p1[i], p2[i])
                yl, yu = xl[i], xu[i]

                rand = np.random.rand()

                # Child 1
                beta = 1.0 + 2.0 * (y1 - yl) / (y2 - y1)
                alpha = 2.0 - beta ** (-(eta_c + 1))
                if rand <= 1.0 / alpha:
                    betaq = (rand * alpha) ** (1.0 / (eta_c + 1))
                else:
                    betaq = (1.0 / (2.0 - rand * alpha)) ** (1.0 / (eta_c + 1))
                c1[i] = 0.5 * (y1 + y2) - 0.5 * betaq * (y2 - y1)

                # Child 2
                beta = 1.0 + 2.0 * (yu - y2) / (y2 - y1)
                alpha = 2.0 - beta ** (-(eta_c + 1))
                if rand <= 1.0 / alpha:
                    betaq = (rand * alpha) ** (1.0 / (eta_c + 1))
                else:
                    betaq = (1.0 / (2.0 - rand * alpha)) ** (1.0 / (eta_c + 1))
                    
                c2[i] = 0.5 * (y1 + y2) + 0.5 * betaq * (y2 - y1)

                # Bound check
                c1[i] = np.clip(c1[i], yl, yu)
                c2[i] = np.clip(c2[i], yl, yu)

    return c1, c2



def polynomial_mutation(x, xl, xu, eta_m=20.0, pm=None):
    """
    Polynomial Mutation.
    Parameters
    ----------
    x      : (n,) decision vector
    xl, xu : (n,) lower / upper bounds
    eta_m  : distribution index (default 20)
    pm     : mutation probability (default 1/n)

    Returns
    -------
    x_mut : (n,) mutated decision vector
    """
    n = len(x)
    if pm is None:
        pm = 1.0 / n

    x_mut = x.copy()
    for i in range(n):
        if np.random.rand() < pm:
            delta1 = (x[i] - xl[i]) / (xu[i] - xl[i])
            delta2 = (xu[i] - x[i]) / (xu[i] - xl[i])
            rand = np.random.rand()
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


