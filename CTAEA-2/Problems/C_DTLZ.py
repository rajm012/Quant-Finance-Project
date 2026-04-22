"""
C-DTLZ Benchmark Suite
=======================
Problems: C1-DTLZ1, C1-DTLZ3, C2-DTLZ2, C3-DTLZ1, C3-DTLZ4
Based on:  Jain & Deb, IEEE TEVC 2014
           Li et al., IEEE TEVC 2019 (C-TAEA paper)

All problems follow the CMOP definition in Equation (1) of the paper.
Constraint violation CV(x) = sum_j max(0, -c_j(x))  [Eq. 3]
"""

import numpy as np


# ---------------------------------------------------------------------------
# Base DTLZ g-functions
# ---------------------------------------------------------------------------

def g_dtlz1(x_m: np.ndarray) -> float:
    """g function for DTLZ1/C1-DTLZ1."""
    k = len(x_m)
    return 100 * (k + np.sum((x_m - 0.5)**2 - np.cos(20 * np.pi * (x_m - 0.5))))


def g_dtlz2(x_m: np.ndarray) -> float:
    """g function for DTLZ2/C2-DTLZ2/C3-DTLZ4."""
    return np.sum((x_m - 0.5)**2)


def g_dtlz3(x_m: np.ndarray) -> float:
    """Same as g_dtlz1 — used for C1-DTLZ3."""
    return g_dtlz1(x_m)


# ---------------------------------------------------------------------------
# Objective functions
# ---------------------------------------------------------------------------

def objectives_dtlz1(x: np.ndarray, m: int) -> np.ndarray:
    """DTLZ1 objective functions."""
    n = len(x)
    k = n - m + 1
    x_m = x[m - 1:]
    g = g_dtlz1(x_m)
    f = np.zeros(m)
    for i in range(m):
        f[i] = 0.5 * (1 + g)
        for j in range(m - 1 - i):
            f[i] *= x[j]
        if i > 0:
            f[i] *= (1 - x[m - 1 - i])
    return f


def objectives_dtlz3(x: np.ndarray, m: int) -> np.ndarray:
    """DTLZ3 objective functions (same structure as DTLZ2 but with g_dtlz3)."""
    n = len(x)
    k = n - m + 1
    x_m = x[m - 1:]
    g = g_dtlz3(x_m)
    f = np.zeros(m)
    for i in range(m):
        f[i] = (1 + g)
        for j in range(m - 1 - i):
            f[i] *= np.cos(x[j] * np.pi / 2)
        if i > 0:
            f[i] *= np.sin(x[m - 1 - i] * np.pi / 2)
    return f


def objectives_dtlz2(x: np.ndarray, m: int) -> np.ndarray:
    """DTLZ2 objective functions."""
    n = len(x)
    k = n - m + 1
    x_m = x[m - 1:]
    g = g_dtlz2(x_m)
    f = np.zeros(m)
    for i in range(m):
        f[i] = (1 + g)
        for j in range(m - 1 - i):
            f[i] *= np.cos(x[j] * np.pi / 2)
        if i > 0:
            f[i] *= np.sin(x[m - 1 - i] * np.pi / 2)
    return f


def objectives_dtlz4(x: np.ndarray, m: int, alpha: float = 100.0) -> np.ndarray:
    """DTLZ4 objective functions (biased mapping via alpha)."""
    n = len(x)
    k = n - m + 1
    x_m = x[m - 1:]
    g = g_dtlz2(x_m)
    f = np.zeros(m)
    for i in range(m):
        f[i] = (1 + g)
        for j in range(m - 1 - i):
            f[i] *= np.cos(x[j]**alpha * np.pi / 2)
        if i > 0:
            f[i] *= np.sin(x[m - 1 - i]**alpha * np.pi / 2)
    return f


# ---------------------------------------------------------------------------
# C1-DTLZ1
# ---------------------------------------------------------------------------

class C1DTLZ1:
    """
    C1-DTLZ1 (Type-1 constrained problem).
    Objective: DTLZ1
    Constraint (Eq. 3 in supplement):
        c(x) = 1 - f_m/0.6 - sum_{i=1}^{m-1} f_i/0.5 >= 0
    """
    name = 'C1-DTLZ1'

    def __init__(self, m: int = 3):
        self.m = m
        self.n_var = m + 4  # k = 5
        self.n_obj = m
        self.n_con = 1
        self.xl = np.zeros(self.n_var)
        self.xu = np.ones(self.n_var)

    def evaluate(self, x: np.ndarray):
        """Returns (F, CV) where F is objective vector, CV is constraint violation."""
        f = objectives_dtlz1(x, self.m)
        # Constraint: c(x) >= 0
        c = 1.0 - (f[-1] / 0.6) - np.sum(f[:-1] / 0.5)
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
        """Sample approximate Pareto front points."""
        from Metrics.reference_points import sample_simplex
        return sample_simplex(self.m, n_points) * 0.5  # PF on hyperplane sum=0.5


# ---------------------------------------------------------------------------
# C1-DTLZ3
# ---------------------------------------------------------------------------

class C1DTLZ3:
    """
    C1-DTLZ3 (Type-1 constrained, infeasible barrier).
    Objective: DTLZ3
    Constraint (Eq. 5 in supplement):
        c(x) = (sum f_i^2 - 16)(sum f_i^2 - r^2) >= 0
    r values: {3:9, 5:12.5, 8:12.5, 10:15, 15:15}
    """
    name = 'C1-DTLZ3'
    R_MAP = {3: 9, 5: 12.5, 8: 12.5, 10: 15, 15: 15}

    def __init__(self, m: int = 3):
        self.m = m
        self.n_var = m + 9   # k = 10
        self.n_obj = m
        self.n_con = 1
        self.r = self.R_MAP.get(m, 9)
        self.xl = np.zeros(self.n_var)
        self.xu = np.ones(self.n_var)

    def evaluate(self, x: np.ndarray):
        f = objectives_dtlz3(x, self.m)
        sum_sq = np.sum(f ** 2)
        c = (sum_sq - 16) * (sum_sq - self.r ** 2)
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
# C2-DTLZ2
# ---------------------------------------------------------------------------

class C2DTLZ2:
    """
    C2-DTLZ2 (Type-2, disjoint feasible regions on PF).
    Objective: DTLZ2
    Constraint (Eq. 7 in supplement), r = 0.1:
        c(x) = max(
            max_i [(f_i - 1)^2 + sum_{j!=i} f_j^2 - r^2],
            [sum_i (f_i - 1/sqrt(m))^2 - r^2]
        ) >= 0
    """
    name = 'C2-DTLZ2'

    def __init__(self, m: int = 3, r: float = 0.1):
        self.m = m
        self.n_var = m + 9  # k = 10
        self.n_obj = m
        self.n_con = 1
        self.r = r
        self.xl = np.zeros(self.n_var)
        self.xu = np.ones(self.n_var)

    def evaluate(self, x: np.ndarray):
        f = objectives_dtlz2(x, self.m)
        r = self.r
        # First term: max over i
        term1 = -np.inf
        for i in range(self.m):
            others_sq = sum(f[j]**2 for j in range(self.m) if j != i)
            val = (f[i] - 1)**2 + others_sq - r**2
            term1 = max(term1, val)
        # Second term
        term2 = np.sum((f - 1.0 / np.sqrt(self.m))**2) - r**2
        c = max(term1, term2)
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
# C3-DTLZ1
# ---------------------------------------------------------------------------

class C3DTLZ1:
    """
    C3-DTLZ1 (Type-3, m constraints).
    Objective: DTLZ1
    Constraints (Eq. 8 in supplement):
        c_j(x) = sum_{i!=j} f_j(x) + f_i(x)/0.5 - 1 >= 0, for j=1..m
    """
    name = 'C3-DTLZ1'

    def __init__(self, m: int = 3):
        self.m = m
        self.n_var = m + 4
        self.n_obj = m
        self.n_con = m
        self.xl = np.zeros(self.n_var)
        self.xu = np.ones(self.n_var)

    def evaluate(self, x: np.ndarray):
        f = objectives_dtlz1(x, self.m)
        cv = 0.0
        for j in range(self.m):
            cj = f[j] + sum(f[i] / 0.5 for i in range(self.m) if i != j) - 1.0
            cv += max(0.0, -cj)
        return f, cv

    def evaluate_batch(self, X: np.ndarray):
        n = len(X)
        F  = np.zeros((n, self.m))
        CV = np.zeros(n)
        for i in range(n):
            F[i], CV[i] = self.evaluate(X[i])
        return F, CV

    def pareto_front_sample(self, n_points: int = 500) -> np.ndarray:
        from Metrics.reference_points import sample_c3dtlz1_pf
        return sample_c3dtlz1_pf(self.m, n_points)


# ---------------------------------------------------------------------------
# C3-DTLZ4
# ---------------------------------------------------------------------------

class C3DTLZ4:
    """
    C3-DTLZ4 (Type-3, m constraints, biased parameter alpha=100).
    Objective: DTLZ4 (alpha=100)
    Constraints (Eq. 11 in supplement):
        c_j(x) = f_j^2/4 + sum_{i!=j} f_i^2 - 1 >= 0, for j=1..m
    """
    name = 'C3-DTLZ4'

    def __init__(self, m: int = 3, alpha: float = 100.0):
        self.m = m
        self.alpha = alpha
        self.n_var = m + 9
        self.n_obj = m
        self.n_con = m
        self.xl = np.zeros(self.n_var)
        self.xu = np.ones(self.n_var)

    def evaluate(self, x: np.ndarray):
        f = objectives_dtlz4(x, self.m, self.alpha)
        cv = 0.0
        for j in range(self.m):
            cj = f[j]**2 / 4.0 + sum(f[i]**2 for i in range(self.m) if i != j) - 1.0
            cv += max(0.0, -cj)
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
