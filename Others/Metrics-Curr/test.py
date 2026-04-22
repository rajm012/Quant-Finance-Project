
import numpy as np
from Problems import C1DTLZ1
from Metrics import calculate_igd, calculate_hypervolume
from Metrics.reference_points import get_reference_points

problem = C1DTLZ1(n_obj=3)
population = np.random.rand(100, problem.n_obj)
ref_points = get_reference_points(problem, n_points=10000)

# Calculate IGD (lower is better)
# Calculate HV (higher is better)

igd = calculate_igd(ref_points, population)
print(f"IGD: {igd:.6f}")
reference_point = np.array([1.1, 1.1, 1.1])
hv = calculate_hypervolume(population, reference_point)
print(f"HV: {hv:.6f}")
