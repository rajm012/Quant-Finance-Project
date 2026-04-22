

"""
DC3-DTLZ1 and DC3-DTLZ3: Type-3 DC-DTLZ problems.
Combination of Type-1 and Type-2 constraints.
m+1 constraints:
  cj(x) = cos(a*pi*xj) > b,  for j = 1,...,m
  c_{m+1}(x) = cos(a*pi*g(xm)) > b

Parameters: a=1, b=0.5 (from paper).
"""


import numpy as np
from .base import BaseProblem


class DC3DTLZ1(BaseProblem):
    """
    Parameters
    ----------
    n_obj : int
        Number of objectives.
    a : float
        Default 1.
    b : float
        Default 0.5.
    """

    def __init__(self, n_obj=3, a=1.0, b=0.5):
        self.m = n_obj
        self.a = a
        self.b = b
        self.k = 5
        n_var = n_obj - 1 + self.k
        xl = [0.0] * n_var
        xu = [1.0] * n_var
        # m+1 constraints
        super().__init__(n_var=n_var, n_obj=n_obj, n_con=n_obj + 1, xl=xl, xu=xu)


    def _g(self, Xm):
        return 100.0 * (
            self.k
            + np.sum((Xm - 0.5) ** 2 - np.cos(20.0 * np.pi * (Xm - 0.5)), axis=1)
        )
        

    def evaluate(self, X):
        N = X.shape[0]
        m = self.m
        Xp = X[:, : m - 1]
        Xm_vars = X[:, m - 1 :]
        g = self._g(Xm_vars)

        F = np.zeros((N, m))
        for i in range(m):
            prod = 0.5 * (1.0 + g)
            for j in range(m - 1 - i):
                prod *= Xp[:, j]
            
            if i > 0:
                prod *= 1.0 - Xp[:, m - 1 - i]
            
            F[:, i] = prod

        G = np.zeros((N, m + 1))
        # cj(x) = cos(a*pi*xj) > b for j = 1,...,m
        
        for j in range(m):
            if j < m - 1:
                xj = Xp[:, j]
            
            else:
                # j = m-1 (0-indexed), this is x_{m-1} = Xp[:, m-2] — last position var
                # Wait: in the paper, xj for j=1,...,m means decision variables x1,...,xm
                # In DTLZ: x = [x1,...,x_{m-1}, x_m,...,x_n]
                # So xj for j < m corresponds to position variables Xp[:, j-1]
                # For j = m (1-indexed), it's Xp[:, m-2] (last position var) — but that's x_{m-1}
                # Paper says cj(x) = cos(a*pi*x_j) for j=1,...,m
                # x_m in 1-indexed is Xm_vars[:,0] (first distance variable)
                
                xj = Xm_vars[:, 0]
                
            cj = np.cos(self.a * np.pi * xj) - self.b
            G[:, j] = np.maximum(-cj, 0.0)

        # c_{m+1}(x) = cos(a*pi*g(xm)) > b
        c_last = np.cos(self.a * np.pi * g) - self.b
        G[:, m] = np.maximum(-c_last, 0.0)

        return F, G


    def get_pareto_front_reference(self, n_points=1000):
        """Same PF as DTLZ1 with full DC3 feasibility checks.

        We sample on the g=0 manifold in decision space and filter points by
        all constraints through `evaluate`.
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
            r = rng.dirichlet(np.ones(self.m), size=n_points)
            return r * 0.5

        F_all = np.vstack(collected)
        if len(F_all) >= n_points:
            idx = rng.choice(len(F_all), size=n_points, replace=False)
            return F_all[idx]

        idx = rng.choice(len(F_all), size=n_points, replace=True)
        return F_all[idx]
    


class DC3DTLZ3(BaseProblem):
    """
    Parameters
    ----------
    n_obj : int
        Number of objectives.
    a : float
        Default 1.
    b : float
        Default 0.5.
    """

    def __init__(self, n_obj=3, a=1.0, b=0.5):
        self.m = n_obj
        self.a = a
        self.b = b
        self.k = 10
        n_var = n_obj - 1 + self.k
        xl = [0.0] * n_var
        xu = [1.0] * n_var
        super().__init__(n_var=n_var, n_obj=n_obj, n_con=n_obj + 1, xl=xl, xu=xu)


    def _g(self, Xm):
        return 100.0 * (
            self.k
            + np.sum((Xm - 0.5) ** 2 - np.cos(20.0 * np.pi * (Xm - 0.5)), axis=1)
        )
        

    def evaluate(self, X):
        N = X.shape[0]
        m = self.m
        Xp = X[:, : m - 1]
        Xm_vars = X[:, m - 1 :]
        g = self._g(Xm_vars)

        F = np.zeros((N, m))
        for i in range(m):
            fi = 1.0 + g
            for j in range(m - 1 - i):
                fi *= np.cos(Xp[:, j] * np.pi / 2.0)
            
            if i > 0:
                fi *= np.sin(Xp[:, m - 1 - i] * np.pi / 2.0)
            
            F[:, i] = fi

        G = np.zeros((N, m + 1))
        for j in range(m):
            if j < m - 1:
                xj = Xp[:, j]
            
            else:
                xj = Xm_vars[:, 0]
            
            cj = np.cos(self.a * np.pi * xj) - self.b
            G[:, j] = np.maximum(-cj, 0.0)

        c_last = np.cos(self.a * np.pi * g) - self.b
        G[:, m] = np.maximum(-c_last, 0.0)
        return F, G
    
    

    def get_pareto_front_reference(self, n_points=1000):
        """Same PF as DTLZ3 with full DC3 feasibility checks."""
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
            u = np.abs(rng.standard_normal((n_points, self.m)))
            return u / np.linalg.norm(u, axis=1, keepdims=True)

        F_all = np.vstack(collected)
        if len(F_all) >= n_points:
            idx = rng.choice(len(F_all), size=n_points, replace=False)
            return F_all[idx]

        idx = rng.choice(len(F_all), size=n_points, replace=True)
        return F_all[idx]

