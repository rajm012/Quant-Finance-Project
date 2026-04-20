import numpy as np
from pymoo.problems import get_problem
from .Base import BaseCMOP

class DC1DTLZ3(BaseCMOP):
    """
    DC1-DTLZ3: DTLZ3 with decision-space constraint:
    cos(3π * x1) > 0.5.
    """
    def __init__(self, n_obj=3):
        super().__init__(n_obj=n_obj)
        self.dtlz3 = get_problem("dtlz3", n_var=self.n_var, n_obj=self.n_obj)

    def _evaluate(self, X, out, *args, **kwargs):
        f = self.dtlz3.evaluate(X)

        c = np.cos(3 * np.pi * X[:, 0]) - 0.5
        out["G"] = -c.reshape(-1, 1)
        out["F"] = f

    def get_pareto_front(self, n_points=10000):
        # DTLZ3 PF (unit sphere)
        points = np.random.rand(n_points, self.n_obj)
        points = points / np.sqrt(np.sum(points**2, axis=1, keepdims=True))
        return points

    
