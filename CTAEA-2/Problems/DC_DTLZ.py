"""
DC-DTLZ Benchmark Suite
========================
Decision-space Constrained DTLZ problems proposed in the C-TAEA paper.
Problems: DC1-DTLZ1, DC1-DTLZ3, DC2-DTLZ1, DC2-DTLZ3, DC3-DTLZ1, DC3-DTLZ3

Constraints act on the DECISION SPACE (unlike C-DTLZ where they act on objective space).
Parameters: a=3, b=0.5 (Type-1, Type-3);  a=1, b=0.5 (Type-2)
"""

import numpy as np
from Problems.C_DTLZ import (
    objectives_dtlz1, objectives_dtlz3,
    g_dtlz1, g_dtlz3
)


# ---------------------------------------------------------------------------
# DC1-DTLZ1  (Type-1: cone-shaped feasible segments on PF)
# ---------------------------------------------------------------------------

class DC1DTLZ1:
    """
    DC1-DTLZ1 — decision-space Type-1 constraints.
    Objectives: DTLZ1
    Constraint (Eq. 12 in supplement):
        c(x) = cos(a*pi*x_1) > b   =>  c(x) - b > 0
    Parameters: a=3, b=0.5
    """
    name = 'DC1-DTLZ1'

    def __init__(self, m: int = 3, a: float = 3.0, b: float = 0.5):
        self.m = m
        self.a = a
        self.b = b
        self.n_var = m + 4
        self.n_obj = m
        self.n_con = 1
        self.xl = np.zeros(self.n_var)
        self.xu = np.ones(self.n_var)

    def evaluate(self, x: np.ndarray):
        f = objectives_dtlz1(x, self.m)
        # c(x) = cos(a*pi*x_1) - b > 0  =>  violation if cos(...) <= b
        c = np.cos(self.a * np.pi * x[0]) - self.b
        cv = max(0.0, -c)
        return f, cv

    def evaluate_batch(self, X: np.ndarray):
        n = len(X)
        F  = np.zeros((n, self.m))
        CV = np.zeros(n)
        for i in range(n):
            F[i], CV[i] = self.evaluate(X[i])
        return F, CV

    def pareto_front_sample(self, n_points: int = 500) -> np.ndarray:
        from Metrics.reference_points import sample_simplex
        return sample_simplex(self.m, n_points) * 0.5


# ---------------------------------------------------------------------------
# DC1-DTLZ3  (Type-1, DTLZ3 base)
# ---------------------------------------------------------------------------

class DC1DTLZ3:
    """
    DC1-DTLZ3 — decision-space Type-1 constraints.
    Objectives: DTLZ3
    Constraint: c(x) = cos(a*pi*x_1) - b > 0
    """
    name = 'DC1-DTLZ3'

    def __init__(self, m: int = 3, a: float = 3.0, b: float = 0.5):
        self.m = m
        self.a = a
        self.b = b
        self.n_var = m + 9
        self.n_obj = m
        self.n_con = 1
        self.xl = np.zeros(self.n_var)
        self.xu = np.ones(self.n_var)

    def evaluate(self, x: np.ndarray):
        f = objectives_dtlz3(x, self.m)
        c = np.cos(self.a * np.pi * x[0]) - self.b
        cv = max(0.0, -c)
        return f, cv

    def evaluate_batch(self, X: np.ndarray):
        n = len(X)
        F  = np.zeros((n, self.m))
        CV = np.zeros(n)
        for i in range(n):
            F[i], CV[i] = self.evaluate(X[i])
        return F, CV

    def pareto_front_sample(self, n_points: int = 500) -> np.ndarray:
        from Metrics.reference_points import sample_sphere_front
        return sample_sphere_front(self.m, n_points, radius=1.0)


# ---------------------------------------------------------------------------
# DC2-DTLZ1  (Type-2: fluctuating CV landscape)
# ---------------------------------------------------------------------------

class DC2DTLZ1:
    """
    DC2-DTLZ1 — decision-space Type-2 constraints (CV fluctuates near PF).
    Objectives: DTLZ1 (same as C1-DTLZ1)
    Two constraints (Eqs. 13-14 in supplement):
        c1(x) = cos(a*pi*g(x_m)) - b > 0
        c2(x) = exp(-g(x_m)) - b > 0
    Parameters: a=1, b=0.5
    """
    name = 'DC2-DTLZ1'

    def __init__(self, m: int = 3, a: float = 1.0, b: float = 0.5):
        self.m = m
        self.a = a
        self.b = b
        self.n_var = m + 4
        self.n_obj = m
        self.n_con = 2
        self.xl = np.zeros(self.n_var)
        self.xu = np.ones(self.n_var)

    def evaluate(self, x: np.ndarray):
        f = objectives_dtlz1(x, self.m)
        x_m = x[self.m - 1:]
        g = g_dtlz1(x_m)

        c1 = np.cos(self.a * np.pi * g) - self.b
        c2 = np.exp(-g) - self.b

        cv = max(0.0, -c1) + max(0.0, -c2)
        return f, cv

    def evaluate_batch(self, X: np.ndarray):
        n = len(X)
        F  = np.zeros((n, self.m))
        CV = np.zeros(n)
        for i in range(n):
            F[i], CV[i] = self.evaluate(X[i])
        return F, CV

    def pareto_front_sample(self, n_points: int = 500) -> np.ndarray:
        from Metrics.reference_points import sample_simplex
        return sample_simplex(self.m, n_points) * 0.5


# ---------------------------------------------------------------------------
# DC2-DTLZ3  (Type-2, DTLZ3 base)
# ---------------------------------------------------------------------------

class DC2DTLZ3:
    """
    DC2-DTLZ3 — decision-space Type-2 constraints.
    Objectives: DTLZ3
    Two constraints:
        c1(x) = cos(a*pi*g(x_m)) - b > 0
        c2(x) = exp(-g(x_m)) - b > 0
    """
    name = 'DC2-DTLZ3'

    def __init__(self, m: int = 3, a: float = 1.0, b: float = 0.5):
        self.m = m
        self.a = a
        self.b = b
        self.n_var = m + 9
        self.n_obj = m
        self.n_con = 2
        self.xl = np.zeros(self.n_var)
        self.xu = np.ones(self.n_var)

    def evaluate(self, x: np.ndarray):
        f = objectives_dtlz3(x, self.m)
        x_m = x[self.m - 1:]
        g = g_dtlz3(x_m)

        c1 = np.cos(self.a * np.pi * g) - self.b
        c2 = np.exp(-g) - self.b

        cv = max(0.0, -c1) + max(0.0, -c2)
        return f, cv

    def evaluate_batch(self, X: np.ndarray):
        n = len(X)
        F  = np.zeros((n, self.m))
        CV = np.zeros(n)
        for i in range(n):
            F[i], CV[i] = self.evaluate(X[i])
        return F, CV

    def pareto_front_sample(self, n_points: int = 500) -> np.ndarray:
        from Metrics.reference_points import sample_sphere_front
        return sample_sphere_front(self.m, n_points, radius=1.0)


# ---------------------------------------------------------------------------
# DC3-DTLZ1  (Type-3: combination of Type-1 + Type-2)
# ---------------------------------------------------------------------------

class DC3DTLZ1:
    """
    DC3-DTLZ1 — decision-space Type-3 constraints (m+1 constraints).
    Objectives: DTLZ1
    Constraints (Eqs. 15-16 in supplement):
        c_j(x) = cos(a*pi*x_j) - b > 0,  for j=1..m
        c_{m+1}(x) = cos(a*pi*g(x_m)) - b > 0
    Parameters: a=3, b=0.5
    """
    name = 'DC3-DTLZ1'

    def __init__(self, m: int = 3, a: float = 3.0, b: float = 0.5):
        self.m = m
        self.a = a
        self.b = b
        self.n_var = m + 4
        self.n_obj = m
        self.n_con = m + 1
        self.xl = np.zeros(self.n_var)
        self.xu = np.ones(self.n_var)

    def evaluate(self, x: np.ndarray):
        f = objectives_dtlz1(x, self.m)
        x_m = x[self.m - 1:]
        g = g_dtlz1(x_m)

        cv = 0.0
        for j in range(self.m):
            cj = np.cos(self.a * np.pi * x[j]) - self.b
            cv += max(0.0, -cj)

        # last constraint on g
        c_last = np.cos(self.a * np.pi * g) - self.b
        cv += max(0.0, -c_last)

        return f, cv

    def evaluate_batch(self, X: np.ndarray):
        n = len(X)
        F  = np.zeros((n, self.m))
        CV = np.zeros(n)
        for i in range(n):
            F[i], CV[i] = self.evaluate(X[i])
        return F, CV

    def pareto_front_sample(self, n_points: int = 500) -> np.ndarray:
        from Metrics.reference_points import sample_simplex
        return sample_simplex(self.m, n_points) * 0.5


# ---------------------------------------------------------------------------
# DC3-DTLZ3  (Type-3, DTLZ3 base)
# ---------------------------------------------------------------------------

class DC3DTLZ3:
    """
    DC3-DTLZ3 — decision-space Type-3 constraints.
    Objectives: DTLZ3
    Constraints (m+1):
        c_j(x) = cos(a*pi*x_j) - b > 0,  for j=1..m
        c_{m+1}(x) = cos(a*pi*g(x_m)) - b > 0
    """
    name = 'DC3-DTLZ3'

    def __init__(self, m: int = 3, a: float = 3.0, b: float = 0.5):
        self.m = m
        self.a = a
        self.b = b
        self.n_var = m + 9
        self.n_obj = m
        self.n_con = m + 1
        self.xl = np.zeros(self.n_var)
        self.xu = np.ones(self.n_var)

    def evaluate(self, x: np.ndarray):
        f = objectives_dtlz3(x, self.m)
        x_m = x[self.m - 1:]
        g = g_dtlz3(x_m)

        cv = 0.0
        for j in range(self.m):
            cj = np.cos(self.a * np.pi * x[j]) - self.b
            cv += max(0.0, -cj)

        c_last = np.cos(self.a * np.pi * g) - self.b
        cv += max(0.0, -c_last)

        return f, cv

    def evaluate_batch(self, X: np.ndarray):
        n = len(X)
        F  = np.zeros((n, self.m))
        CV = np.zeros(n)
        for i in range(n):
            F[i], CV[i] = self.evaluate(X[i])
        return F, CV

    def pareto_front_sample(self, n_points: int = 500) -> np.ndarray:
        from Metrics.reference_points import sample_sphere_front
        return sample_sphere_front(self.m, n_points, radius=1.0)
