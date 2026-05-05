

"""
DC1-DTLZ1 and DC1-DTLZ3: Type-1 DC-DTLZ problems.
Constraints act on the DECISION SPACE.
c(x) = cos(a*pi*x1) > b  (several infeasible cone-shaped segments on PF)

Parameters: a=3, b=0.5 (as used in paper).
"""


import numpy as np
from .base import BaseProblem


class DC1DTLZ1(BaseProblem):
    """
    Parameters
    ----------
    n_obj : int
        Number of objectives.
    a : float
        Controls number of feasible segments. Default 3.
    b : float
        Controls size of feasible segments. Default 0.5.
    """

    def __init__(self, n_obj=3, a=3.0, b=0.5):
        self.m = n_obj
        self.a = a
        self.b = b
        self.k = 5
        n_var = n_obj - 1 + self.k
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
        for i in range(m):
            prod = 0.5 * (1.0 + g)
            for j in range(m - 1 - i):
                prod *= Xp[:, j]
            if i > 0:
                prod *= 1.0 - Xp[:, m - 1 - i]
            F[:, i] = prod

        # c(x) = cos(a*pi*x1) > b  => feasible if cos(a*pi*x1) > b
        c = np.cos(self.a * np.pi * X[:, 0]) - self.b
        G = np.maximum(-c, 0.0)[:, np.newaxis]

        return F, G


    def get_pareto_front_reference(self, n_points=1000):
        rng = np.random.default_rng(42)
        pts = []
        while len(pts) < n_points:
            r = rng.dirichlet(np.ones(self.m), size=2000)
            f = r * 0.5
            
            # Recover x1 from f: f[-1] = 0.5*(1-x1)*(1+g_min)
            # On PF, g=0, so f[-1] = 0.5*(1-x1) => x1 = 1 - 2*f[-1]
            # But PF: sum fi = 0.5, x1 = 1 - 2*f[m-1]/1 is wrong for m>2
            # Simpler: x1 = 1 - 2*f[:,-1] for the boundary
            # Actually x1 in DTLZ1 maps via product, use x1 from PF samples:
            # f1/f2/.../fm are such that sum=0.5 and x1=1-2*f_m
            # For general m, x1 is the last factor in the product formula
            # We can parameterize: x1 = 1 - fm/0.5  (only valid for m=2)
            # For simplicity, use cos check via f ratios
            # f_{m} = 0.5*(1-x1), so x1 = 1 - 2*f[:,m-1]
            
            x1 = 1.0 - 2.0 * f[:, -1]
            c = np.cos(self.a * np.pi * x1) - self.b
            feasible = (c > 0) & (x1 >= 0) & (x1 <= 1)
            pts.extend(f[feasible].tolist())
        
        return np.array(pts[:n_points])



class DC1DTLZ3(BaseProblem):
    """
    Parameters
    ----------
    n_obj : int
        Number of objectives.
    a : float
        Controls number of feasible segments. Default 3.
    b : float
        Controls size of feasible segments. Default 0.5.
    """


    def __init__(self, n_obj=3, a=3.0, b=0.5):
        self.m = n_obj
        self.a = a
        self.b = b
        self.k = 10
        n_var = n_obj - 1 + self.k
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
        for i in range(m):
            fi = 1.0 + g
            for j in range(m - 1 - i):
                fi *= np.cos(Xp[:, j] * np.pi / 2.0)
            
            if i > 0:
                fi *= np.sin(Xp[:, m - 1 - i] * np.pi / 2.0)
            
            F[:, i] = fi

        c = np.cos(self.a * np.pi * X[:, 0]) - self.b
        G = np.maximum(-c, 0.0)[:, np.newaxis]

        return F, G
    

    def get_pareto_front_reference(self, n_points=1000):
        rng = np.random.default_rng(42)
        pts = []
        while len(pts) < n_points:
            # Sample on unit hypersphere quadrant
            
            u = np.abs(rng.standard_normal((2000, self.m)))
            norms = np.linalg.norm(u, axis=1, keepdims=True)
            f = u / norms
            
            # x1 comes from the angular parameterization:
            # f_m = sin(x1*pi/2) => x1 = (2/pi)*arcsin(f_m)
            
            x1 = (2.0 / np.pi) * np.arcsin(np.clip(f[:, -1], 0, 1))
            c = np.cos(self.a * np.pi * x1) - self.b
            feasible = (c > 0)
            pts.extend(f[feasible].tolist())
            
        return np.array(pts[:n_points])

