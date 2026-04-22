import numpy as np
from Others.Problems import *

def get_reference_points(problem, n_points=10000):
    """
    Get reference points for IGD calculation.
    For problems with known analytical PF, sample from it.
    For others, sample feasible solutions and keep nondominated.
    """
    problem_name = problem.__class__.__name__
    
    if problem_name in ["C1DTLZ1", "C3DTLZ1"]:
        # PF: sum f_i = 0.5, f_i >= 0
        points = np.random.rand(n_points, problem.n_obj)
        points = points / np.sum(points, axis=1, keepdims=True) * 0.5
        return points
    
    elif problem_name in ["C1DTLZ3", "C2DTLZ2", "C3DTLZ4"]:
        # PF: unit sphere quadrant (f_i >= 0, sum f_i^2 = 1)
        points = np.random.rand(n_points, problem.n_obj)
        points = points / np.sqrt(np.sum(points**2, axis=1, keepdims=True))
        return points
    
    elif problem_name.startswith("DC"):
        # For DC problems, PF is same as baseline DTLZ
        # But only portions are feasible. Sample and filter.
        points = []
        while len(points) < n_points:
            cand = np.random.rand(1, problem.n_obj)
            if problem_name in ["DC1DTLZ1", "DC3DTLZ1", "DC2DTLZ1"]:
                # DTLZ1 PF
                cand = cand / np.sum(cand) * 0.5
            else:
                # DTLZ3 PF
                cand = cand / np.sqrt(np.sum(cand**2))
            points.append(cand.flatten())
        return np.array(points)
    
    else:
        raise ValueError(f"No reference points defined for {problem_name}")
    
    