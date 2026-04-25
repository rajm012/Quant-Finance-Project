

"""
C3-DTLZ4: Type-3 constrained problem (DTLZ4 variant with C3 constraints).
Biases solution density; PF is formed by constraint surfaces.

cj(x) = fj^2/4 + sum_{i!=j} fi^2 - 1 >= 0,  for j = 1,...,m
alpha = 100 (DTLZ4 parameter)
"""


import numpy as np
from .base import BaseProblem


class C3DTLZ4(BaseProblem):
    """
    Parameters
    ----------
    n_obj : int
        Number of objectives.
    alpha : float
        Bias parameter (default 100).
    """

    def __init__(self, n_obj=3, alpha=100.0):
        self.m = n_obj
        self.alpha = alpha
        self.k = 10
        n_var = n_obj - 1 + self.k
        xl = [0.0] * n_var
        xu = [1.0] * n_var
        super().__init__(n_var=n_var, n_obj=n_obj, n_con=n_obj, xl=xl, xu=xu)


    def _g(self, Xm):
        return np.sum((Xm - 0.5) ** 2, axis=1)


    def evaluate(self, X):
        N = X.shape[0]
        m = self.m
        alpha = self.alpha
        Xp = X[:, : m - 1]
        Xm = X[:, m - 1 :]
        g = self._g(Xm)

        F = np.zeros((N, m))
        # DTLZ4: use x^alpha instead of x
        
        for i in range(m):
            fi = 1.0 + g
            for j in range(m - 1 - i):
                fi *= np.cos((Xp[:, j] ** alpha) * np.pi / 2.0)
            
            if i > 0:
                fi *= np.sin((Xp[:, m - 1 - i] ** alpha) * np.pi / 2.0)
            
            F[:, i] = fi

        # cj(x) = fj^2/4 + sum_{i!=j} fi^2 - 1 >= 0
        G = np.zeros((N, m))
        for j in range(m):
            cj = F[:, j] ** 2 / 4.0
            for i in range(m):
                if i != j:
                    cj = cj + F[:, i] ** 2
            
            cj = cj - 1.0
            G[:, j] = np.maximum(-cj, 0.0)

        return F, G
    

    def get_pareto_front_reference(self, n_points=1000):
        """Sample feasible PF reference points without infinite loops.

        Candidates are generated on the DTLZ4 PF manifold (g=0) in decision
        space and filtered through the same constraints as `evaluate`.
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
            u = np.abs(rng.standard_normal((n_points, self.m)))
            return u / np.linalg.norm(u, axis=1, keepdims=True)

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

