
import numpy as np
from pymoo.problems import get_problem
from .Base import BaseCMOP

class DC2DTLZ3(BaseCMOP):
    """
    DC2-DTLZ3: DTLZ3 with two constraints on g(xm):
    cos(3π * g) > 0.5  and  exp(-g) > 0.5.
    """
    def __init__(self, n_obj=3):
        super().__init__(n_obj=n_obj)
        self.dtlz3 = get_problem("dtlz3", n_var=self.n_var, n_obj=self.n_obj)

    def _g(self, X):
        # DTLZ3 g function (same as DTLZ1)
        xm = X[:, self.n_obj-1:]
        k = xm.shape[1]
        sum1 = np.sum((xm - 0.5)**2 - np.cos(20 * np.pi * (xm - 0.5)), axis=1)
        return 100 * (k + sum1)

    def _evaluate(self, X, out, *args, **kwargs):
        f = self.dtlz3.evaluate(X)

        g = self._g(X)
        c1 = np.cos(3 * np.pi * g) - 0.5
        c2 = np.exp(-g) - 0.5
        G = np.column_stack([-c1, -c2])
        out["G"] = G
        out["F"] = f

    def get_pareto_front(self, n_points=10000):
        # DTLZ3 PF (unit sphere)
        points = np.random.rand(n_points, self.n_obj)
        points = points / np.sqrt(np.sum(points**2, axis=1, keepdims=True))
        return points

    