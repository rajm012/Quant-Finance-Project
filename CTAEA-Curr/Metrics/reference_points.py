

"""
Reference point generation utilities for sampling the Pareto front.
Used for IGD computation.
"""


import numpy as np
# from itertools import combinations_with_replacement


def generate_weight_vectors(m, H):
    """
    Generate uniformly distributed weight vectors on a canonical simplex
    using the Das-Dennis method.

    Parameters
    ----------
    m : int
        Number of objectives.
    H : int
        Number of divisions per objective.

    Returns
    -------
    W : np.ndarray, shape (N, m)
        Weight vectors, each row sums to 1.
    """
    
    def _generate(m, H):
        if m == 1:
            return [[H]]
        
        result = []
        for i in range(H + 1):
            for rest in _generate(m - 1, H - i):
                result.append([i] + rest)
        return result

    W = np.array(_generate(m, H), dtype=float)
    W = W / H
    return W



def get_reference_point_count(m):
    """
    Get the recommended H value and weight vector count for given m objectives,
    matching Table IV from the paper.
    Parameters
    ----------
    m : int

    Returns
    -------
    N : int
        Number of weight vectors (population size proxy).
    H : int
        Division parameter.
    """
    
    # From Table IV: m -> (N, H or combined)
    table = {
        3: (91, 12),
        5: (210, 6),
        8: (156, 4),  # two-layer
        10: (275, 4),
        15: (135, 3),
    }
    return table.get(m, (100, 5))



def sample_pareto_front_simplex(m, n_points):
    """
    Sample points uniformly on the canonical simplex (for DTLZ1-type PFs).
    Parameters
    ----------
    m : int
        Number of objectives.
    n_points : int

    Returns
    -------
    P_star : np.ndarray, shape (n_points, m)
        Sum of each row = 0.5 (DTLZ1 PF).
    """
    rng = np.random.default_rng(42)
    pts = rng.dirichlet(np.ones(m), size=n_points)
    return pts * 0.5



def sample_pareto_front_sphere(m, n_points):
    """
    Sample points uniformly on the unit hypersphere quadrant (for DTLZ2/3/4-type PFs).
    Parameters
    ----------
    m : int
        Number of objectives.
    n_points : int

    Returns
    -------
    P_star : np.ndarray, shape (n_points, m)
    """
    
    rng = np.random.default_rng(42)
    u = np.abs(rng.standard_normal((n_points * 10, m)))
    norms = np.linalg.norm(u, axis=1, keepdims=True)
    pts = u / norms
    
    # Only positive quadrant
    pts = pts[np.all(pts >= 0, axis=1)]
    return pts[:n_points]



def get_pf_reference_points(problem_name, m, n_points=500):
    """
    Get reference points for the PF of a given problem.
    Parameters
    ----------
    problem_name : str
        Name of the problem (e.g., 'C1DTLZ1').
    m : int
        Number of objectives.
    n_points : int
        Number of reference points.

    Returns
    -------
    P_star : np.ndarray
    """
    
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
    
    problem_map = {
        'C1DTLZ1': ('Problems.C1DTLZ1', 'C1DTLZ1'),
        'C1DTLZ3': ('Problems.C1DTLZ3', 'C1DTLZ3'),
        'C2DTLZ2': ('Problems.C2DTLZ2', 'C2DTLZ2'),
        'C3DTLZ1': ('Problems.C3DTLZ1', 'C3DTLZ1'),
        'C3DTLZ4': ('Problems.C3DTLZ4', 'C3DTLZ4'),
        'DC1DTLZ1': ('Problems.DC1DTLZ1', 'DC1DTLZ1'),
        'DC1DTLZ3': ('Problems.DC1DTLZ1', 'DC1DTLZ3'),
        'DC2DTLZ1': ('Problems.DC2DTLZ1', 'DC2DTLZ1'),
        'DC2DTLZ3': ('Problems.DC2DTLZ1', 'DC2DTLZ3'),
        'DC3DTLZ1': ('Problems.DC3DTLZ1', 'DC3DTLZ1'),
        'DC3DTLZ3': ('Problems.DC3DTLZ1', 'DC3DTLZ3'),
    }

    if problem_name not in problem_map:
        raise ValueError(f"Unknown problem: {problem_name}")

    mod_name, cls_name = problem_map[problem_name]
    import importlib
    mod = importlib.import_module(mod_name)
    cls = getattr(mod, cls_name)
    prob = cls(n_obj=m)
    return prob.get_pareto_front_reference(n_points=n_points)


