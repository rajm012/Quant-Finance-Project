

"""
DC2-DTLZ1 and DC2-DTLZ3: Type-2 DC-DTLZ problems.
Constraints act on the decision space via g(xm).
The CV of a solution fluctuates when converging to the PF, creating local optima.

Two constraints:
  c1(x) = cos(a*pi*g(xm)) > b
  c2(x) = exp(-g(xm)) > b

Parameters: a=1, b=0.5 (default; paper uses various settings).
"""


import numpy as np
from .base import BaseProblem


class DC2DTLZ1(BaseProblem):
    """
    Parameters
    ----------
    n_obj : int
        Number of objectives.
    a : float
        Controls CV oscillation frequency. Default 1.
    b : float
        Controls CV local optima height. Default 0.5.
    """

    def __init__(self, n_obj=3, a=1.0, b=0.5):
        self.m = n_obj
        self.a = a
        self.b = b
        self.k = 5
        n_var = n_obj - 1 + self.k
        xl = [0.0] * n_var
        xu = [1.0] * n_var
        super().__init__(n_var=n_var, n_obj=n_obj, n_con=2, xl=xl, xu=xu)


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

        # c1(x) = cos(a*pi*g) > b
        c1 = np.cos(self.a * np.pi * g) - self.b
        
        # c2(x) = exp(-g) > b
        c2 = np.exp(-g) - self.b

        G = np.zeros((N, 2))
        G[:, 0] = np.maximum(-c1, 0.0)
        G[:, 1] = np.maximum(-c2, 0.0)
        return F, G
    
    

    def get_pareto_front_reference(self, n_points=1000):
        """
        PF is the same as DTLZ1: sum fi = 0.5 (when g=0).
        At g=0: c1 = cos(0) = 1 > b and c2 = exp(0) = 1 > b => feasible.
        """
        
        rng = np.random.default_rng(42)
        r = rng.dirichlet(np.ones(self.m), size=n_points)
        return r * 0.5




class DC2DTLZ3(BaseProblem):
    """
    Parameters
    ----------
    n_obj : int
        Number of objectives.
    a : float
        Controls CV oscillation frequency. Default 1.
    b : float
        Controls CV local optima height. Default 0.5.
    """

    def __init__(self, n_obj=3, a=1.0, b=0.5):
        self.m = n_obj
        self.a = a
        self.b = b
        self.k = 10
        n_var = n_obj - 1 + self.k
        xl = [0.0] * n_var
        xu = [1.0] * n_var
        super().__init__(n_var=n_var, n_obj=n_obj, n_con=2, xl=xl, xu=xu)


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

        c1 = np.cos(self.a * np.pi * g) - self.b
        c2 = np.exp(-g) - self.b
        G = np.zeros((N, 2))
        G[:, 0] = np.maximum(-c1, 0.0)
        G[:, 1] = np.maximum(-c2, 0.0)
        return F, G
    

    def get_pareto_front_reference(self, n_points=1000):
        rng = np.random.default_rng(42)
        u = np.abs(rng.standard_normal((n_points, self.m)))
        norms = np.linalg.norm(u, axis=1, keepdims=True)
        return u / norms

