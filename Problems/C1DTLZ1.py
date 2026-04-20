
# pyright: reportGeneralTypeIssues=false

import numpy as np
from pymoo.problems import get_problem
from .Base import BaseCMOP

class C1DTLZ1(BaseCMOP):
    """
    C1-DTLZ1: DTLZ1 with constraint 1 - f_m/0.6 - sum_{i=1}^{m-1}(f_i/0.5) >= 0.
    """
    def __init__(self, n_obj=3):
        super().__init__(n_obj=n_obj)
        # Use pymoo's DTLZ1 for objective calculation
        self.dtlz1 = get_problem("dtlz1", n_var=self.n_var, n_obj=self.n_obj)

    def _evaluate(self, X, out, *args, **kwargs):
        # Compute DTLZ1 objectives
        f = self.dtlz1.evaluate(X)

        # Compute constraint violation
        sum_f_except_last = np.sum(f[:, :-1] / 0.5, axis=1)     # type: ignore
        c = 1 - (f[:, -1] / 0.6) - sum_f_except_last            # type: ignore
        # pymoo expects G <= 0
        out["G"] = -c.reshape(-1, 1)
        out["F"] = f

    def get_pareto_front(self, n_points=10000):
        # DTLZ1 PF: sum f_i = 0.5, f_i >= 0
        points = np.random.rand(n_points, self.n_obj)
        points = points / np.sum(points, axis=1, keepdims=True) * 0.5
        return points

