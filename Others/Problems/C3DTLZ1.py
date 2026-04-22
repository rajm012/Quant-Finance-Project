
# pyright: reportGeneralTypeIssues=false

import numpy as np
from pymoo.problems import get_problem
from .Base import BaseCMOP

class C3DTLZ1(BaseCMOP):
    """
    C3-DTLZ1: DTLZ1 with m constraints:
    sum_{i != j} f_i / 0.5 + f_j - 1 >= 0, for j = 1..m.
    """
    def __init__(self, n_obj=3):
        super().__init__(n_obj=n_obj)
        self.dtlz1 = get_problem("dtlz1", n_var=self.n_var, n_obj=self.n_obj)

    def _evaluate(self, X, out, *args, **kwargs):
        f = self.dtlz1.evaluate(X)

        G = np.zeros((X.shape[0], self.n_obj))
        for j in range(self.n_obj):
            sum_others = np.sum(f, axis=1) - f[:, j]              # type: ignore
            c = sum_others / 0.5 + f[:, j] - 1                    # type: ignore
            G[:, j] = -c  # pymoo expects G <= 0
        out["G"] = G
        out["F"] = f

    def get_pareto_front(self, n_points=10000):
        # Same as DTLZ1 PF
        points = np.random.rand(n_points, self.n_obj)
        points = points / np.sum(points, axis=1, keepdims=True) * 0.5
        return points
    
