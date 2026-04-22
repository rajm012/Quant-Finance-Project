
# pyright: reportGeneralTypeIssues=false

import numpy as np
from pymoo.problems import get_problem
from .Base import BaseCMOP

class C3DTLZ4(BaseCMOP):
    """
    C3-DTLZ4: DTLZ4 (alpha=100) with constraints:
    f_j^2 / 4 + sum_{i != j} f_i^2 - 1 >= 0, for j = 1..m.
    """
    def __init__(self, n_obj=3, alpha=100):
        super().__init__(n_obj=n_obj)
        self.dtlz4 = get_problem("dtlz4", n_var=self.n_var, n_obj=self.n_obj, alpha=alpha)

    def _evaluate(self, X, out, *args, **kwargs):
        f = self.dtlz4.evaluate(X)

        sum_sq = np.sum(f**2, axis=1)             # type: ignore
        G = np.zeros((X.shape[0], self.n_obj))
        for j in range(self.n_obj):
            c = (f[:, j]**2) / 4 + (sum_sq - f[:, j]**2) - 1          # type: ignore
            G[:, j] = -c
        out["G"] = G
        out["F"] = f


    def get_pareto_front(self, n_points=10000):
        # Sample from DTLZ4 PF (unit sphere) and keep feasible points
        points = []
        while len(points) < n_points:
            cand = np.random.rand(1, self.n_obj)
            cand = cand / np.sqrt(np.sum(cand**2))
            feasible = True
            for j in range(self.n_obj):
                c = (cand[0, j]**2) / 4 + (1 - cand[0, j]**2) - 1
                if c < 0:
                    feasible = False
                    break
            if feasible:
                points.append(cand.flatten())
        return np.array(points[:n_points])
    
    