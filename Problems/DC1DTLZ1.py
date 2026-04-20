
import numpy as np
from pymoo.problems import get_problem
from .Base import BaseCMOP

class DC1DTLZ1(BaseCMOP):
    """
    DC1-DTLZ1: DTLZ1 with decision-space constraint:
    cos(3π * x1) > 0.5.
    """
    def __init__(self, n_obj=3):
        super().__init__(n_obj=n_obj)
        self.dtlz1 = get_problem("dtlz1", n_var=self.n_var, n_obj=self.n_obj)

    def _evaluate(self, X, out, *args, **kwargs):
        # f = self.dtlz1.evaluate(X)
        f = self.dtlz1.evaluate(X, return_values_of=["F"])

        # Constraint on first decision variable
        c = np.cos(3 * np.pi * X[:, 0]) - 0.5
        out["G"] = -c.reshape(-1, 1)
        out["F"] = f

    def get_pareto_front(self, n_points=10000):
        # For DC1-DTLZ1, the PF is the same as DTLZ1 PF
        points = np.random.rand(n_points, self.n_obj)
        points = points / np.sum(points, axis=1, keepdims=True) * 0.5
        return points
   
   