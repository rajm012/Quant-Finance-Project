

"""
Inverted Generational Distance (IGD) metric.
IGD(P, P*) = (1/|P*|) * sum_{z in P*} dist(z, P)
where dist(z, P) is the Euclidean distance from z to its nearest neighbor in P.
"""


import numpy as np


def igd(P, P_star):
    """
    Compute Inverted Generational Distance.
    Parameters
    ----------
    P : np.ndarray, shape (N, m)
        Approximation set (obtained solutions).
    P_star : np.ndarray, shape (K, m)
        Reference set (true Pareto front samples).

    Returns
    -------
    float
        IGD value. Smaller is better.
    """
    
    if len(P) == 0:
        return np.inf

    P = np.array(P)
    P_star = np.array(P_star)

    # For each point in P_star, find min distance to P
    dists = np.zeros(len(P_star))
    for i, z in enumerate(P_star):
        diff = P - z  # (N, m)
        dists[i] = np.min(np.sqrt(np.sum(diff ** 2, axis=1)))

    return np.mean(dists)



def igd_plus(P, P_star):
    """
    Compute IGD+ (modified IGD).
    Parameters
    ----------
    P : np.ndarray, shape (N, m)
    P_star : np.ndarray, shape (K, m)

    Returns
    -------
    float
        IGD+ value. Smaller is better.
    """
    if len(P) == 0:
        return np.inf

    P = np.array(P)
    P_star = np.array(P_star)
    dists = np.zeros(len(P_star))
    
    for i, z in enumerate(P_star):
        diff = np.maximum(P - z, 0.0)  # (N, m)
        dists[i] = np.min(np.sqrt(np.sum(diff ** 2, axis=1)))

    return np.mean(dists)


