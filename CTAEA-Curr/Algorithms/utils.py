

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
    Fast non-dominated sorting (NSGA-II style), vectorized dominance matrix.
    Parameters
    ----------
    F : (N, m) objective values

    Returns
    -------
    fronts : list of lists
        fronts[0] = list of indices in first non-dominated front, etc.
    rank   : (N,) rank of each individual (0 = first front)
    """
    F = np.asarray(F, dtype=float)
    N = F.shape[0]
    if N == 0:
        return [], np.array([], dtype=int)
    if N == 1:
        return [[0]], np.zeros(1, dtype=int)

    le = (F[:, None, :] <= F[None, :, :]).all(axis=2)
    lt = (F[:, None, :] < F[None, :, :]).any(axis=2)
    dom = le & lt
    np.fill_diagonal(dom, False)

    n_dom = dom.sum(axis=0).astype(np.int32)
    rank = -np.ones(N, dtype=np.int32)
    fronts = []
    assigned = np.zeros(N, dtype=bool)
    r = 0
    current = np.where((n_dom == 0) & (~assigned))[0]

    while len(current) > 0:
        fronts.append(current.tolist())
        assigned[current] = True
        rank[current] = r
        n_dom = n_dom - dom[current, :].sum(axis=0).astype(np.int32)
        r += 1
        current = np.where((n_dom == 0) & (~assigned))[0]

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
    """Return boolean mask of non-dominated solutions (vectorized, minimization)."""
    F = np.asarray(F, dtype=float)
    N = F.shape[0]
    if N <= 1:
        return np.ones(N, dtype=bool)

    # dom[j, i] <=> solution j dominates solution i
    le = (F[:, None, :] <= F[None, :, :]).all(axis=2)
    lt = (F[:, None, :] < F[None, :, :]).any(axis=2)
    dom = le & lt
    np.fill_diagonal(dom, False)
    dominated = dom.any(axis=0)
    return ~dominated


def constrained_dominance_matrix(F, CV):
    """
    cd[p, q] is True iff p constrained-dominates q (Jain & Deb style for CMOPs).

    Rules: feasible beats infeasible; among infeasible, lower CV wins;
    among feasible solutions, use Pareto dominance on objectives.
    """
    F = np.asarray(F, dtype=float)
    CV = np.asarray(CV, dtype=float).reshape(-1)
    N = F.shape[0]
    if N <= 1:
        return np.zeros((N, N), dtype=bool)

    feas = CV == 0.0
    dom_feas = feas[:, None] & (~feas[None, :])
    both_inf = (~feas[:, None]) & (~feas[None, :])
    cv_dom = both_inf & (CV[:, None] < CV[None, :])
    both_feas = feas[:, None] & feas[None, :]
    le = (F[:, None, :] <= F[None, :, :]).all(axis=2)
    lt = (F[:, None, :] < F[None, :, :]).any(axis=2)
    pareto = le & lt & both_feas
    np.fill_diagonal(pareto, False)
    cd = dom_feas | cv_dom | pareto
    np.fill_diagonal(cd, False)
    return cd


def fast_constrained_non_dominated_sort(F, CV):
    """
    Non-dominated sorting under constrained dominance (vectorized).
    Same front semantics as the nested-loop NSGA-II variant on constrained ranks.
    """
    F = np.asarray(F, dtype=float)
    CV = np.asarray(CV, dtype=float).reshape(-1)
    N = F.shape[0]
    if N == 0:
        return [], np.array([], dtype=int)
    if N == 1:
        return [[0]], np.zeros(1, dtype=int)

    cd = constrained_dominance_matrix(F, CV)
    n_dom = cd.sum(axis=0).astype(np.int32)
    rank = -np.ones(N, dtype=np.int32)
    fronts = []
    assigned = np.zeros(N, dtype=bool)
    r = 0
    current = np.where((n_dom == 0) & (~assigned))[0]

    while len(current) > 0:
        fronts.append(current.tolist())
        assigned[current] = True
        rank[current] = r
        n_dom = n_dom - cd[current, :].sum(axis=0).astype(np.int32)
        r += 1
        current = np.where((n_dom == 0) & (~assigned))[0]

    return fronts, rank



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


def sbx_crossover_batch(p1, p2, xl, xu, eta_c=30.0, pc=0.9):
    """
    SBX for many parent pairs at once (fully vectorized over pairs and variables).

    Note: consumes random numbers in a different order than calling ``sbx_crossover``
    row-by-row, so results for a fixed seed will not match the scalar loop — only the
    statistical operator is the same.
    """
    p1 = np.asarray(p1, dtype=float)
    p2 = np.asarray(p2, dtype=float)
    xl = np.asarray(xl, dtype=float)
    xu = np.asarray(xu, dtype=float)
    K, n = p1.shape
    c1, c2 = p1.copy(), p2.copy()
    skip = np.random.rand(K) > pc
    rand_dim = np.random.rand(K, n)
    rand_u = np.random.rand(K, n)
    y1 = np.minimum(p1, p2)
    y2 = np.maximum(p1, p2)
    diff_ok = np.abs(p1 - p2) > 1e-14
    active = (~skip[:, None]) & (rand_dim <= 0.5) & diff_ok
    span = np.where(active, y2 - y1, 1.0)
    beta1 = 1.0 + 2.0 * (y1 - xl) / span
    alpha1 = 2.0 - np.power(beta1, -(eta_c + 1))
    inva1 = 1.0 / alpha1
    betaq1 = np.where(
        rand_u <= inva1,
        np.power(rand_u * alpha1, 1.0 / (eta_c + 1)),
        np.power(1.0 / (2.0 - rand_u * alpha1), 1.0 / (eta_c + 1)),
    )
    child1_dim = 0.5 * (y1 + y2) - 0.5 * betaq1 * (y2 - y1)
    beta2 = 1.0 + 2.0 * (xu - y2) / span
    alpha2 = 2.0 - np.power(beta2, -(eta_c + 1))
    inva2 = 1.0 / alpha2
    betaq2 = np.where(
        rand_u <= inva2,
        np.power(rand_u * alpha2, 1.0 / (eta_c + 1)),
        np.power(1.0 / (2.0 - rand_u * alpha2), 1.0 / (eta_c + 1)),
    )
    child2_dim = 0.5 * (y1 + y2) + 0.5 * betaq2 * (y2 - y1)
    nc1 = np.clip(child1_dim, xl, xu)
    nc2 = np.clip(child2_dim, xl, xu)
    c1[:] = np.where(active, nc1, p1)
    c2[:] = np.where(active, nc2, p2)
    return c1, c2


def polynomial_mutation_batch(X, xl, xu, eta_m=20.0, pm=None):
    """
    Polynomial mutation for a population ``X`` of shape (K, n_var).

    Same RNG-order caveat as ``sbx_crossover_batch`` vs repeated scalar calls.
    """
    X = np.asarray(X, dtype=float)
    xl = np.asarray(xl, dtype=float)
    xu = np.asarray(xu, dtype=float)
    K, n = X.shape
    if pm is None:
        pm = 1.0 / n
    mutate = np.random.rand(K, n) < pm
    rand = np.random.rand(K, n)
    delta1 = (X - xl) / (xu - xl)
    delta2 = (xu - X) / (xu - xl)
    mut_pow = 1.0 / (eta_m + 1.0)
    half = rand < 0.5
    xy1 = 1.0 - delta1
    xy2 = 1.0 - delta2
    val1 = 2.0 * rand + (1.0 - 2.0 * rand) * np.power(xy1, eta_m + 1)
    val2 = 2.0 * (1.0 - rand) + 2.0 * (rand - 0.5) * np.power(xy2, eta_m + 1)
    deltaq = np.where(half, np.power(val1, mut_pow) - 1.0, 1.0 - np.power(val2, mut_pow))
    new_x = X + deltaq * (xu - xl)
    new_x = np.clip(new_x, xl, xu)
    return np.where(mutate, new_x, X)


def sbx_crossover_match_scalar_rng(p1, p2, xl, xu, eta_c=30.0, pc=0.9):
    """
    Apply ``sbx_crossover`` row-wise to ``p1``, ``p2`` of shape (K, n).

    Random number consumption matches ``K`` sequential calls to ``sbx_crossover``
    in row order (pair 0, then pair 1, ...).
    """
    p1 = np.asarray(p1, dtype=float)
    p2 = np.asarray(p2, dtype=float)
    K = p1.shape[0]
    c1 = np.empty_like(p1)
    c2 = np.empty_like(p2)
    for k in range(K):
        a, b = sbx_crossover(p1[k], p2[k], xl, xu, eta_c, pc)
        c1[k], c2[k] = a, b
    return c1, c2


def polynomial_mutation_match_scalar_rng(X, xl, xu, eta_m=20.0, pm=None):
    """
    Apply ``polynomial_mutation`` row-wise; RNG order matches ``K`` scalar calls.
    """
    X = np.asarray(X, dtype=float)
    out = np.empty_like(X)
    for k in range(X.shape[0]):
        out[k] = polynomial_mutation(X[k], xl, xu, eta_m, pm)
    return out


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


