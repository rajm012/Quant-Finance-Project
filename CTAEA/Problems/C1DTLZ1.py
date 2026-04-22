
"""
C1-DTLZ1: Type-1 constrained problem.
The original PF is kept the same, but there is an infeasible barrier
that causes difficulties for an algorithm in converging toward the PF.
"""

import numpy as np
from .base import BaseProblem
# from itertools import combinations


class C1DTLZ1(BaseProblem):
    """
    Parameters
    ----------
    n_obj : int
        Number of objectives (m). Supported: 3, 5, 8, 10, 15.
    """

    def __init__(self, n_obj=3):
        self.m = n_obj
        # k = n_var - n_obj + 1, typically k=5 for DTLZ1
        self.k = 5
        n_var = n_obj - 1 + self.k
        xl = [0.0] * n_var
        xu = [1.0] * n_var
        super().__init__(n_var=n_var, n_obj=n_obj, n_con=1, xl=xl, xu=xu)

    def _g(self, Xm):
        # Xm shape: (N, k)
        return 100.0 * (
            self.k
            + np.sum((Xm - 0.5) ** 2 - np.cos(20.0 * np.pi * (Xm - 0.5)), axis=1)
        )

    def evaluate(self, X):
        N = X.shape[0]
        m = self.m
        k = self.k
        Xp = X[:, : m - 1]   # (N, m-1)  position variables
        Xm = X[:, m - 1 :]   # (N, k)    distance variables
        g = self._g(Xm)  # (N,)
        F = np.zeros((N, m))
        
        # Compute objectives
        # f1 = 0.5 * x1*x2*...*x_{m-1} * (1+g)
        # f2 = 0.5 * x1*x2*...*(1-x_{m-1})*(1+g)
        # ...
        # fm = 0.5 * (1-x1)*(1+g)
        
        for i in range(m):
            prod = 0.5 * (1.0 + g)
            for j in range(m - 1 - i):
                prod *= Xp[:, j]
            
            if i > 0:
                prod *= 1.0 - Xp[:, m - 1 - i]
            
            F[:, i] = prod

        # Constraint: c(x) = 1 - fm/0.6 - sum_{i=1}^{m-1} fi/0.5 >= 0
        # Violation form: G > 0 means infeasible
        
        c = 1.0 - F[:, m - 1] / 0.6 - np.sum(F[:, : m - 1] / 0.5, axis=1)
        G = -c[:, np.newaxis]  # G > 0 => violated
        return F, G

    def get_pareto_front_reference(self, n_points=1000):
        """Approximate PF reference points on the feasible PF."""
        # PF of DTLZ1 is the hyperplane sum(fi) = 0.5, fi >= 0
        # We sample uniformly on this simplex
        
        rng = np.random.default_rng(42)
        pts = []
        while len(pts) < n_points:
            r = rng.dirichlet(np.ones(self.m))
            f = r * 0.5
            # Check feasibility: 1 - fm/0.6 - sum(fi[:-1])/0.5 >= 0
            c = 1.0 - f[-1] / 0.6 - np.sum(f[:-1] / 0.5)
            if c >= 0:
                pts.append(f)
                
        return np.array(pts[:n_points])

