
"""
C1-DTLZ3: Type-1 constrained problem with an infeasible barrier intersecting
the attainable objective space.
The constraint is:
  c(x) = (sum fi^2 - 16)(sum fi^2 - r^2) >= 0
where r is set per m as recommended in Jain & Deb 2014.
"""

import numpy as np
from .base import BaseProblem


class C1DTLZ3(BaseProblem):
    """
    C1-DTLZ3 problem.
    Parameters
    ----------
    n_obj : int
        Number of objectives. Supported: 3, 5, 8, 10, 15.
    """

    # r values as recommended: {3:9, 5:12.5, 8:12.5, 10:15, 15:15}
    R_VALUES = {3: 9.0, 5: 12.5, 8: 12.5, 10: 15.0, 15: 15.0}

    def __init__(self, n_obj=3):
        self.m = n_obj
        self.k = 10  # DTLZ3 uses k=10
        n_var = n_obj - 1 + self.k
        self.r = self.R_VALUES.get(n_obj, 9.0)
        xl = [0.0] * n_var
        xu = [1.0] * n_var
        super().__init__(n_var=n_var, n_obj=n_obj, n_con=1, xl=xl, xu=xu)


    def _g(self, Xm):
        return 100.0 * (
            self.k
            + np.sum((Xm - 0.5) ** 2 - np.cos(20.0 * np.pi * (Xm - 0.5)), axis=1)
        )


    def evaluate(self, X):
        N = X.shape[0]
        m = self.m
        Xp = X[:, : m - 1]
        Xm = X[:, m - 1 :]
        g = self._g(Xm)

        F = np.zeros((N, m))
        # DTLZ3 objectives (spherical mapping)
        for i in range(m):
            fi = 1.0 + g
            for j in range(m - 1 - i):
                fi *= np.cos(Xp[:, j] * np.pi / 2.0)
            
            if i > 0:
                fi *= np.sin(Xp[:, m - 1 - i] * np.pi / 2.0)
            
            F[:, i] = fi

        # Constraint: (sum fi^2 - 16)(sum fi^2 - r^2) >= 0
        sum_f2 = np.sum(F ** 2, axis=1)
        c = (sum_f2 - 16.0) * (sum_f2 - self.r ** 2)
        G = -c[:, np.newaxis]  # G > 0 => violated
        return F, G
    

    def get_pareto_front_reference(self, n_points=1000):
        """
        The PF of DTLZ3 lies on the unit hypersphere (sum fi^2 = 1), fi >= 0.
        We sample uniformly on that spherical simplex and keep feasible points.
        """
        rng = np.random.default_rng(42)
        pts = []
        while len(pts) < n_points:
            # Sample on positive orthant of unit sphere
            u = np.abs(rng.standard_normal((100, self.m)))
            norms = np.linalg.norm(u, axis=1, keepdims=True)
            f = u / norms
            sum_f2 = np.sum(f ** 2, axis=1)
            c = (sum_f2 - 16.0) * (sum_f2 - self.r ** 2)
            feasible = c >= 0
            pts.extend(f[feasible].tolist())
            
        return np.array(pts[:n_points])

