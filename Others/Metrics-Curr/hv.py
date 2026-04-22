
import numpy as np
from pymoo.indicators.hv import HV as PyMOO_HV

def calculate_hypervolume(population, reference_point):
    """
    Calculate Hypervolume (HV).
    
    Parameters:
    -----------
    population : np.ndarray
        Feasible solutions obtained from algorithm (shape: n_solutions × n_obj)
    reference_point : np.ndarray
        Worst point dominated by all Pareto optimal vectors (shape: n_obj,)
        Paper uses: (1.1, 1.1, ..., 1.1) for most problems
                    (2.1, 2.1, ..., 2.1) for C3-DTLZ4
    Returns:
    --------
    hv : float
        Hypervolume value (higher is better)
    """
    
    # Remove solutions dominated by reference point (paper requirement)
    # A solution is dominated by reference if all objectives <= reference
    # But reference is worst point (large values), so solutions should be <= reference
    # Actually: reference is a point that is WORSE than all Pareto optimal vectors
    # So all feasible solutions should dominate the reference point
    # No need to filter, but we ensure reference is an array
    # PyMOO's HV expects minimization problems
    # For maximization, we can negate or use offset
    # Since our problems are minimization, we can use directly
    
    if len(population) == 0:
        return 0.0
    
    if population.ndim == 1:
        population = population.reshape(1, -1)
        
    mask = np.all(population <= reference_point, axis=1)
    filtered_pop = population[mask]
    
    if len(filtered_pop) == 0:
        return 0.0

    hv_indicator = PyMOO_HV(ref_point=reference_point)
    hv = hv_indicator(filtered_pop)
    return hv



def calculate_hypervolume_custom(population, reference_point):
    """
    Custom hypervolume calculation (2D only) - for verification.
    For higher dimensions, use PyMOO.
    """
    if len(population) == 0 or population.shape[1] != 2:
        raise ValueError("Custom HV only works for 2D problems")
    
    sorted_pop = population[np.argsort(population[:, 0])]
    hv = 0.0
    prev_f1 = reference_point[0]
    for i in range(len(sorted_pop)):
        f1 = sorted_pop[i, 0]
        f2 = sorted_pop[i, 1]
        
        # Contribution of this point
        if f1 < reference_point[0] and f2 < reference_point[1]:
            width = min(prev_f1, reference_point[0]) - max(f1, 0)
            height = reference_point[1] - f2
            hv += width * height
            prev_f1 = f1
    
    return max(0, hv)


def normalize_objectives(f, ideal=None, nadir=None):
    """
    Normalize objectives to [0, 1] range for HV calculation.
    Paper normalizes before HV calculation (Section V-D).
    """
    
    if ideal is None:
        ideal = np.min(f, axis=0)
        
    if nadir is None:
        nadir = np.max(f, axis=0)
    
    range_f = nadir - ideal
    range_f[range_f == 0] = 1.0    
    normalized = (f - ideal) / range_f
    return normalized, ideal, nadir


# need to see while implementation
def normalize_to_pf(f, problem_name):
    """
    Normalizes based on the known True PF bounds mentioned in the paper.
    """
    # Define nadir based on problem type
    if "DTLZ1" in problem_name:
        nadir = np.array([0.5] * f.shape[1])
        
    elif "DTLZ4" in problem_name and "C3" in problem_name:
        nadir = np.array([2.0] * f.shape[1]) # C3-DTLZ4 has a larger scale
        
    else:
        nadir = np.array([1.0] * f.shape[1])
    
    ideal = np.zeros(f.shape[1])
    normalized = (f - ideal) / (nadir - ideal + 1e-10)
    return normalized

