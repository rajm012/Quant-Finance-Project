
# not to be used now

import numpy as np

def filter_feasible(F, G):
    """
    Keep only feasible solutions.
    Parameters
    ----------
    F : ndarray (N * m)
        Objective values
    G : ndarray (N * k)
        Constraint violations

    Returns
    -------
    F_feasible : ndarray
    """

    if F is None or len(F) == 0:
        return np.empty((0, 0))

    if G is None:
        return F

    feasible_mask = np.all(G <= 0, axis=1)
    return F[feasible_mask]


def normalize_objectives(F, ideal=None, nadir=None):
    """
    Normalize objectives to [0, 1].
    Required before HV calculation.
    """

    if len(F) == 0:
        return F, None, None

    if ideal is None:
        ideal = np.min(F, axis=0)

    if nadir is None:
        nadir = np.max(F, axis=0)

    diff = nadir - ideal
    diff[diff == 0] = 1.0
    F_norm = (F - ideal) / diff
    return F_norm, ideal, nadir

