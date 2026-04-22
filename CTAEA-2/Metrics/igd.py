"""
Inverted Generational Distance (IGD) metric.
IGD(P, P*) = sum_{z in P*} dist(z, P) / |P*|   (Eq. 14 in paper)
"""

import numpy as np


def igd(P: np.ndarray, P_star: np.ndarray) -> float:
    """
    Compute IGD value.

    Parameters
    ----------
    P      : ndarray (n, m) — solutions obtained by the algorithm (feasible only)
    P_star : ndarray (k, m) — reference points uniformly sampled from true PF

    Returns
    -------
    float : IGD value (lower is better)
    """
    if len(P) == 0:
        return np.inf

    # For each z in P*, compute min distance to P
    # Using broadcasting for efficiency
    # dist_matrix[i, j] = ||P_star[i] - P[j]||
    diff = P_star[:, np.newaxis, :] - P[np.newaxis, :, :]  # (k, n, m)
    dists = np.linalg.norm(diff, axis=2)                    # (k, n)
    min_dists = np.min(dists, axis=1)                       # (k,)

    return np.mean(min_dists)


def igd_plus(P: np.ndarray, P_star: np.ndarray) -> float:
    """
    IGD+ metric (Ishibuchi et al. 2015) — a weakly Pareto-compliant IGD variant.
    dist+(z, P) = min_{y in P} sqrt(sum_i max(y_i - z_i, 0)^2)
    """
    if len(P) == 0:
        return np.inf

    dists = []
    for z in P_star:
        d_vals = np.sqrt(np.sum(np.maximum(P - z, 0) ** 2, axis=1))
        dists.append(np.min(d_vals))
    return np.mean(dists)
