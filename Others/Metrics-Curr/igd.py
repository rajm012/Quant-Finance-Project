

"""
Note: 
    when on any model we are testing then when calling the IGD, try calling it on only feasible solutions only.
"""


import numpy as np
from scipy.spatial import cKDTree       # type: ignore

def calculate_igd(pareto_front, population):
    """
    Calculate Inverted Generational Distance (IGD).
    Parameters:
    -----------
    pareto_front : np.ndarray
        Reference points uniformly sampled along the true PF (shape: n_points * n_obj)
    population : np.ndarray
        Solutions obtained from algorithm (shape: n_solutions * n_obj)
    
    Returns:
    --------
    igd : float
        IGD value (lower is better)
    """
    
    # assuming that only feasible are passed
    
    if len(population) == 0:
        return np.inf

    tree = cKDTree(population)
    distances, _ = tree.query(pareto_front)
    igd = np.mean(distances)
    return igd


def calculate_igd_from_files(ref_points_path, population_path):
    """Helper to load from files and compute IGD"""
    ref = np.loadtxt(ref_points_path)
    pop = np.loadtxt(population_path)
    return calculate_igd(ref, pop)

