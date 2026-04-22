
# pyright: reportGeneralTypeIssues=false

import numpy as np
from pymoo.problems import get_problem
from .Base import BaseCMOP

class C1DTLZ3(BaseCMOP):
    """
    C1-DTLZ3: DTLZ3 with constraint (sum f_i^2 - 16) * (sum f_i^2 - r^2) >= 0.
    r depends on number of objectives (see supplement).
    """
    def __init__(self, n_obj=3):
        super().__init__(n_obj=n_obj)
        self.dtlz3 = get_problem("dtlz3", n_var=self.n_var, n_obj=self.n_obj)
        # r values from supplement (m: r)
        self.r_map = {3:9, 5:12.5, 8:12.5, 10:15, 15:15}
        self.r = self.r_map.get(n_obj, 12.5)

    def _evaluate(self, X, out, *args, **kwargs):
        f = self.dtlz3.evaluate(X)

        sum_sq = np.sum(f**2, axis=1)                     # type: ignore
        c = (sum_sq - 16) * (sum_sq - self.r**2)
        out["G"] = -c.reshape(-1, 1)
        out["F"] = f

    def get_pareto_front(self, n_points=10000):
        # DTLZ3 PF: unit sphere quadrant
        points = np.random.rand(n_points, self.n_obj)
        points = points / np.sqrt(np.sum(points**2, axis=1, keepdims=True))
        return points
    
    