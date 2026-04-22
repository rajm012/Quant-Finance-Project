

"""
C2-DTLZ2: Type-2 constrained problem.
Constraint introduces infeasibility to some parts of the PF (disjoint feasible segments along the PF).
r = 0.1 as used in the paper.
"""


import numpy as np
from .base import BaseProblem


class C2DTLZ2(BaseProblem):
    """
    Parameters
    ----------
    n_obj : int
        Number of objectives.
    r : float
        Radius parameter. Default 0.1 (as used in C-TAEA paper).
    """

    def __init__(self, n_obj=3, r=0.1):
        self.m = n_obj
        self.r = r
        self.k = 10  # DTLZ2 uses k=10
        n_var = n_obj - 1 + self.k
        xl = [0.0] * n_var
        xu = [1.0] * n_var
        super().__init__(n_var=n_var, n_obj=n_obj, n_con=1, xl=xl, xu=xu)


    def _g(self, Xm):
        return np.sum((Xm - 0.5) ** 2, axis=1)


    def evaluate(self, X):
        N = X.shape[0]
        m = self.m
        Xp = X[:, : m - 1]
        Xm = X[:, m - 1 :]
        g = self._g(Xm)
        F = np.zeros((N, m))
        
        # DTLZ2 objectives (spherical mapping)
        for i in range(m):
            fi = 1.0 + g
            for j in range(m - 1 - i):
                fi *= np.cos(Xp[:, j] * np.pi / 2.0)
            
            if i > 0:
                fi *= np.sin(Xp[:, m - 1 - i] * np.pi / 2.0)
            
            F[:, i] = fi

        r = self.r
        
        # c(x) = max(
        #    max_i[ (fi-1)^2 + sum_{j!=i} fj^2 - r^2 ],
        #    sum_i[ (fi - 1/sqrt(m))^2 ] - r^2
        # ) >= 0
        # Violation: G > 0 means infeasible, so G = -c
        
        term1 = np.full(N, -np.inf)
        for i in range(m):
            val = (F[:, i] - 1.0) ** 2
            for j in range(m):
                if j != i:
                    val = val + F[:, j] ** 2
            
            val = val - r ** 2
            term1 = np.maximum(term1, val)

        term2 = np.sum((F - 1.0 / np.sqrt(m)) ** 2, axis=1) - r ** 2
        c = np.maximum(term1, term2)  # c >= 0 means feasible
        G = c[:, np.newaxis]  # G > 0 => violated (c > 0 means violated here)

        # Re-check: constraint is c(x) >= 0, feasible when c >= 0
        # So violated when c < 0 => G_violation = max(-c, 0)
        G = np.maximum(-c, 0.0)[:, np.newaxis]

        return F, G


    def get_pareto_front_reference(self, n_points=1000):
        """
        PF of DTLZ2 is the unit hypersphere quadrant.
        Keep only feasible points under C2 constraint.
        """
        rng = np.random.default_rng(42)
        pts = []
        while len(pts) < n_points:
            u = np.abs(rng.standard_normal((500, self.m)))
            norms = np.linalg.norm(u, axis=1, keepdims=True)
            f = u / norms

            r = self.r
            term1 = np.full(len(f), -np.inf)
            for i in range(self.m):
                val = (f[:, i] - 1.0) ** 2
                for j in range(self.m):
                    if j != i:
                        val = val + f[:, j] ** 2
                
                val = val - r ** 2
                term1 = np.maximum(term1, val)
            
            term2 = np.sum((f - 1.0 / np.sqrt(self.m)) ** 2, axis=1) - r ** 2
            c = np.maximum(term1, term2)
            feasible = c >= 0
            pts.extend(f[feasible].tolist())
        
        return np.array(pts[:n_points])

