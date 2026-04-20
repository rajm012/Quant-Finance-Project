
import numpy as np
from pymoo.problems import get_problem
from .Base import BaseCMOP

class DC3DTLZ1(BaseCMOP):
    """
    DC3-DTLZ1: DTLZ1 with m+1 constraints:
    For j=1..m: cos(3π * x_j) > 0.5
    and cos(3π * g) > 0.5.
    """
    def __init__(self, n_obj=3):
        super().__init__(n_obj=n_obj)
        self.dtlz1 = get_problem("dtlz1", n_var=self.n_var, n_obj=self.n_obj)

    def _g(self, X):
        xm = X[:, self.n_obj-1:]
        k = xm.shape[1]
        sum1 = np.sum((xm - 0.5)**2 - np.cos(20 * np.pi * (xm - 0.5)), axis=1)
        return 100 * (k + sum1)

    def _evaluate(self, X, out, *args, **kwargs):
        f = self.dtlz1.evaluate(X)

        # Constraints on first m decision variables
        cons = []
        for j in range(min(self.n_obj, self.n_var)):
            c = np.cos(3 * np.pi * X[:, j]) - 0.5
            cons.append(-c)
        # Constraint on g
        g = self._g(X)
        c_g = np.cos(3 * np.pi * g) - 0.5
        cons.append(-c_g)
        out["G"] = np.column_stack(cons)
        out["F"] = f

    def get_pareto_front(self, n_points=10000):
        # DTLZ1 PF
        points = np.random.rand(n_points, self.n_obj)
        points = points / np.sum(points, axis=1, keepdims=True) * 0.5
        return points

    
