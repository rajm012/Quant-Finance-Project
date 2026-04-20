
# pyright: reportGeneralTypeIssues=false

import numpy as np
from pymoo.problems import get_problem
from .Base import BaseCMOP

class C2DTLZ2(BaseCMOP):
    """
    C2-DTLZ2: DTLZ2 with constraint
    max( max_i ((f_i-1)^2 + sum_{j!=i} f_j^2 - r^2), sum_i (f_i - 1/sqrt(m))^2 - r^2 ) >= 0.
    r = 0.1.
    """
    def __init__(self, n_obj=3):
        super().__init__(n_obj=n_obj)
        self.dtlz2 = get_problem("dtlz2", n_var=self.n_var, n_obj=self.n_obj)
        self.r = 0.1

    def _evaluate(self, X, out, *args, **kwargs):
        f = self.dtlz2.evaluate(X)

        sum_sq = np.sum(f**2, axis=1)             # type: ignore
        # First part: max_i ((f_i-1)^2 + sum_{j!=i} f_j^2 - r^2)
        term1 = np.zeros(X.shape[0])
        for i in range(self.n_obj):
            term_i = (f[:, i] - 1)**2 + (sum_sq - f[:, i]**2) - self.r**2         # type: ignore
            term1 = np.maximum(term1, term_i)
        # Second part: sum_i (f_i - 1/sqrt(m))^2 - r^2
        term2 = np.sum((f - 1/np.sqrt(self.n_obj))**2, axis=1) - self.r**2
        c = np.maximum(term1, term2)
        out["G"] = c.reshape(-1, 1)
        out["F"] = f

    def get_pareto_front(self, n_points=10000):
        # Sample from DTLZ2 PF and keep points satisfying constraint
        points = []
        while len(points) < n_points:
            cand = np.random.rand(1, self.n_obj)
            cand = cand / np.sqrt(np.sum(cand**2))
            # Check constraint (with g=0)
            sum_sq = 1.0
            term1 = -np.inf
            for i in range(self.n_obj):
                term_i = (cand[0, i] - 1)**2 + (1 - cand[0, i]**2) - self.r**2
                term1 = max(term1, term_i)
            term2 = np.sum((cand - 1/np.sqrt(self.n_obj))**2) - self.r**2
            c = max(term1, term2)
            if c >= 0:
                points.append(cand.flatten())
        return np.array(points[:n_points])
    
    
    