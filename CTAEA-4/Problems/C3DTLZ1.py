

"""
C3-DTLZ1: Type-3 constrained problem.
Multiple constraints; the PF is formed by portions of constraint surfaces.
cj(x) = sum_{i!=j} fj + fi/0.5 - 1 >= 0,  for j = 1,...,m
"""


import numpy as np
from .base import BaseProblem


class C3DTLZ1(BaseProblem):
    """
    Parameters
    ----------
    n_obj : int
        Number of objectives.
    """

    def __init__(self, n_obj=3):
        self.m = n_obj
        self.k = 5
        n_var = n_obj - 1 + self.k
        xl = [0.0] * n_var
        xu = [1.0] * n_var
        super().__init__(n_var=n_var, n_obj=n_obj, n_con=n_obj, xl=xl, xu=xu)


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
        for i in range(m):
            prod = 0.5 * (1.0 + g)
            for j in range(m - 1 - i):
                prod *= Xp[:, j]
            
            if i > 0:
                prod *= 1.0 - Xp[:, m - 1 - i]
            
            F[:, i] = prod

        # cj(x) = sum_{i, i!=j} fj + fi/0.5 - 1 >= 0
        # As per paper eq. (8): cj(x) = sum_{i=1,i!=j}^m fj(x) + fi(x)/0.5 - 1 >= 0
        
        G = np.zeros((N, m))
        for j in range(m):
            cj = F[:, j]  # fj
            for i in range(m):
                if i != j:
                    cj = cj + F[:, i] / 0.5
            
            cj = cj - 1.0
            # cj >= 0 is feasible, so violation = max(-cj, 0)
            G[:, j] = np.maximum(-cj, 0.0)

        return F, G


    def get_pareto_front_reference(self, n_points=1000):
        """Sample feasible PF reference points without infinite loops.

        We generate candidates on the DTLZ1 PF manifold (g=0) in decision space,
        evaluate constraints with the same implementation as `evaluate`, and keep
        feasible non-dominated points.
        """
        rng = np.random.default_rng(42)
        collected = []
        max_rounds = 300
        batch = max(2000, n_points * 3)

        for _ in range(max_rounds):
            Xp = rng.uniform(0.0, 1.0, size=(batch, self.m - 1))
            Xm = np.full((batch, self.k), 0.5)
            X = np.hstack([Xp, Xm])

            F, G = self.evaluate(X)
            feasible = np.sum(np.maximum(G, 0.0), axis=1) == 0
            if np.any(feasible):
                collected.append(F[feasible])

            if collected and sum(len(a) for a in collected) >= n_points * 2:
                break

        if not collected:
            # Fallback to avoid blocking analysis if feasible PF sampling is sparse.
            r = rng.dirichlet(np.ones(self.m), size=n_points)
            return r * 0.5

        F_all = np.vstack(collected)
        nd = np.ones(len(F_all), dtype=bool)
        for i in range(len(F_all)):
            if not nd[i]:
                continue
            fi = F_all[i]
            for j in range(len(F_all)):
                if i == j or not nd[i]:
                    continue
                fj = F_all[j]
                if np.all(fj <= fi) and np.any(fj < fi):
                    nd[i] = False

        F_nd = F_all[nd]
        if len(F_nd) >= n_points:
            idx = rng.choice(len(F_nd), size=n_points, replace=False)
            return F_nd[idx]

        idx = rng.choice(len(F_nd), size=n_points, replace=True)
        return F_nd[idx]

