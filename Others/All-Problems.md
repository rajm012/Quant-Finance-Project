
# Combined Problems Implementation
# All code from the Problems folder (latest version)

# ================= Base.py =================

from pymoo.core.problem import Problem

class BaseCMOP(Problem):
    """
    Base class for Constrained Multi-Objective Problems.
    All problems in this paper inherit from this class.
    """
    def __init__(self, n_var=None, n_obj=3, xl=0, xu=1, **kwargs):
        # Default n_var = n_obj + 4 (standard for DTLZ problems)
        if n_var is None:
            n_var = n_obj + 4
        super().__init__(n_var=n_var, n_obj=n_obj, xl=xl, xu=xu, **kwargs)

    def _evaluate(self, X, out, *args, **kwargs):
        """
        This method must be implemented by subclasses.
        It should compute the objective values and constraint violations.
        """
        raise NotImplementedError

    def get_pareto_front(self, n_points=10000):
        """
        Return the true Pareto front points (for IGD calculation).
        This method should be implemented by each problem.
        """
        raise NotImplementedError
    
    
# ================= C1DTLZ1.py =================

# pyright: reportGeneralTypeIssues=false

import numpy as np
from pymoo.problems import get_problem
from .Base import BaseCMOP

class C1DTLZ1(BaseCMOP):
    """
    C1-DTLZ1: DTLZ1 with constraint 1 - f_m/0.6 - sum_{i=1}^{m-1}(f_i/0.5) >= 0.
    """
    def __init__(self, n_obj=3):
        super().__init__(n_obj=n_obj)
        # Use pymoo's DTLZ1 for objective calculation
        self.dtlz1 = get_problem("dtlz1", n_var=self.n_var, n_obj=self.n_obj)

    def _evaluate(self, X, out, *args, **kwargs):
        # Compute DTLZ1 objectives
        f = self.dtlz1.evaluate(X)

        # Compute constraint violation
        sum_f_except_last = np.sum(f[:, :-1] / 0.5, axis=1)     # type: ignore
        c = 1 - (f[:, -1] / 0.6) - sum_f_except_last            # type: ignore
        # pymoo expects G <= 0
        out["G"] = -c.reshape(-1, 1)
        out["F"] = f

    def get_pareto_front(self, n_points=10000):
        # DTLZ1 PF: sum f_i = 0.5, f_i >= 0
        points = np.random.rand(n_points, self.n_obj)
        points = points / np.sum(points, axis=1, keepdims=True) * 0.5
        return points


# ================= C1DTLZ3.py =================

# pyright: reportGeneralTypeIssues=false

import numpy as np
from pymoo.problems import get_problem
from .Base import BaseCMOP

class C1DTLZ3(BaseCMOP):
    """
    C1-DTLZ3: DTLZ3 with constraint (sum f_i^2 - 16) * (sum f_i^2 - r^2) >= 0.
    r depends on number of objectives (see supplement).
    """
    def __init__(self, n_obj=3):
        super().__init__(n_obj=n_obj)
        self.dtlz3 = get_problem("dtlz3", n_var=self.n_var, n_obj=self.n_obj)
        # r values from supplement (m: r)
        self.r_map = {3:9, 5:12.5, 8:12.5, 10:15, 15:15}
        self.r = self.r_map.get(n_obj, 12.5)

    def _evaluate(self, X, out, *args, **kwargs):
        f = self.dtlz3.evaluate(X)

        sum_sq = np.sum(f**2, axis=1)                     # type: ignore
        c = (sum_sq - 16) * (sum_sq - self.r**2)
        out["G"] = -c.reshape(-1, 1)
        out["F"] = f

    def get_pareto_front(self, n_points=10000):
        # DTLZ3 PF: unit sphere quadrant
        points = np.random.rand(n_points, self.n_obj)
        points = points / np.sqrt(np.sum(points**2, axis=1, keepdims=True))
        return points
    

# ================= C2DTLZ2.py =================

# pyright: reportGeneralTypeIssues=false

import numpy as np
from pymoo.problems import get_problem
from .Base import BaseCMOP

class C2DTLZ2(BaseCMOP):
    """
    C2-DTLZ2: DTLZ2 with constraint
    max( max_i ((f_i-1)^2 + sum_{j!=i} f_j^2 - r^2), sum_i (f_i - 1/sqrt(m))^2 - r^2 ) >= 0.
    r = 0.1.
    """
    def __init__(self, n_obj=3):
        super().__init__(n_obj=n_obj)
        self.dtlz2 = get_problem("dtlz2", n_var=self.n_var, n_obj=self.n_obj)
        self.r = 0.1

    def _evaluate(self, X, out, *args, **kwargs):
        f = self.dtlz2.evaluate(X)

        sum_sq = np.sum(f**2, axis=1)             # type: ignore
        # First part: max_i ((f_i-1)^2 + sum_{j!=i} f_j^2 - r^2)
        term1 = np.zeros(X.shape[0])
        for i in range(self.n_obj):
            term_i = (f[:, i] - 1)**2 + (sum_sq - f[:, i]**2) - self.r**2         # type: ignore
            term1 = np.maximum(term1, term_i)
        # Second part: sum_i (f_i - 1/sqrt(m))^2 - r^2
        term2 = np.sum((f - 1/np.sqrt(self.n_obj))**2, axis=1) - self.r**2
        c = np.maximum(term1, term2)
        out["G"] = -c.reshape(-1, 1)
        out["F"] = f

    def get_pareto_front(self, n_points=10000):
        # Sample from DTLZ2 PF and keep points satisfying constraint
        points = []
        while len(points) < n_points:
            cand = np.random.rand(1, self.n_obj)
            cand = cand / np.sqrt(np.sum(cand**2))
            # Check constraint (with g=0)
            sum_sq = 1.0
            term1 = -np.inf
            for i in range(self.n_obj):
                term_i = (cand[0, i] - 1)**2 + (1 - cand[0, i]**2) - self.r**2
                term1 = max(term1, term_i)
            term2 = np.sum((cand - 1/np.sqrt(self.n_obj))**2) - self.r**2
            c = max(term1, term2)
            if c >= 0:
                points.append(cand.flatten())
        return np.array(points[:n_points])
    

# ================= C3DTLZ1.py =================

# pyright: reportGeneralTypeIssues=false

import numpy as np
from pymoo.problems import get_problem
from .Base import BaseCMOP

class C3DTLZ1(BaseCMOP):
    """
    C3-DTLZ1: DTLZ1 with m constraints:
    sum_{i != j} f_i / 0.5 + f_j - 1 >= 0, for j = 1..m.
    """
    def __init__(self, n_obj=3):
        super().__init__(n_obj=n_obj)
        self.dtlz1 = get_problem("dtlz1", n_var=self.n_var, n_obj=self.n_obj)

    def _evaluate(self, X, out, *args, **kwargs):
        f = self.dtlz1.evaluate(X)

        G = np.zeros((X.shape[0], self.n_obj))
        for j in range(self.n_obj):
            sum_others = np.sum(f, axis=1) - f[:, j]              # type: ignore
            c = sum_others / 0.5 + f[:, j] - 1                    # type: ignore
            G[:, j] = -c  # pymoo expects G <= 0
        out["G"] = G
        out["F"] = f

    def get_pareto_front(self, n_points=10000):
        # Same as DTLZ1 PF
        points = np.random.rand(n_points, self.n_obj)
        points = points / np.sum(points, axis=1, keepdims=True) * 0.5
        return points
    

# ================= C3DTLZ4.py =================

# pyright: reportGeneralTypeIssues=false

import numpy as np
from pymoo.problems import get_problem
from .Base import BaseCMOP

class C3DTLZ4(BaseCMOP):
    """
    C3-DTLZ4: DTLZ4 (alpha=100) with constraints:
    f_j^2 / 4 + sum_{i != j} f_i^2 - 1 >= 0, for j = 1..m.
    """
    def __init__(self, n_obj=3, alpha=100):
        super().__init__(n_obj=n_obj)
        self.dtlz4 = get_problem("dtlz4", n_var=self.n_var, n_obj=self.n_obj, alpha=alpha)

    def _evaluate(self, X, out, *args, **kwargs):
        f = self.dtlz4.evaluate(X)

        sum_sq = np.sum(f**2, axis=1)             # type: ignore
        G = np.zeros((X.shape[0], self.n_obj))
        for j in range(self.n_obj):
            c = (f[:, j]**2) / 4 + (sum_sq - f[:, j]**2) - 1          # type: ignore
            G[:, j] = -c
        out["G"] = G
        out["F"] = f


    def get_pareto_front(self, n_points=10000):
        # Sample from DTLZ4 PF (unit sphere) and keep feasible points
        points = []
        while len(points) < n_points:
            cand = np.random.rand(1, self.n_obj)
            cand = cand / np.sqrt(np.sum(cand**2))
            feasible = True
            for j in range(self.n_obj):
                c = (cand[0, j]**2) / 4 + (1 - cand[0, j]**2) - 1
                if c < 0:
                    feasible = False
                    break
            if feasible:
                points.append(cand.flatten())
        return np.array(points[:n_points])
     

# ================= DC1DTLZ1.py =================

import numpy as np
from pymoo.problems import get_problem
from .Base import BaseCMOP

class DC1DTLZ1(BaseCMOP):
    """
    DC1-DTLZ1: DTLZ1 with decision-space constraint:
    cos(3π * x1) > 0.5.
    """
    def __init__(self, n_obj=3):
        super().__init__(n_obj=n_obj)
        self.dtlz1 = get_problem("dtlz1", n_var=self.n_var, n_obj=self.n_obj)

    def _evaluate(self, X, out, *args, **kwargs):
        f = self.dtlz1.evaluate(X)

        # Constraint on first decision variable
        c = np.cos(3 * np.pi * X[:, 0]) - 0.5
        out["G"] = -c.reshape(-1, 1)
        out["F"] = f

    def get_pareto_front(self, n_points=10000):
        # For DC1-DTLZ1, the PF is the same as DTLZ1 PF
        points = np.random.rand(n_points, self.n_obj)
        points = points / np.sum(points, axis=1, keepdims=True) * 0.5
        return points
    

# ================= DC1DTLZ3.py =================
import numpy as np
from pymoo.problems import get_problem
from .Base import BaseCMOP

class DC1DTLZ3(BaseCMOP):
    """
    DC1-DTLZ3: DTLZ3 with decision-space constraint:
    cos(3π * x1) > 0.5.
    """
    def __init__(self, n_obj=3):
        super().__init__(n_obj=n_obj)
        self.dtlz3 = get_problem("dtlz3", n_var=self.n_var, n_obj=self.n_obj)

    def _evaluate(self, X, out, *args, **kwargs):
        f = self.dtlz3.evaluate(X)

        c = np.cos(3 * np.pi * X[:, 0]) - 0.5
        out["G"] = -c.reshape(-1, 1)
        out["F"] = f

    def get_pareto_front(self, n_points=10000):
        # DTLZ3 PF (unit sphere)
        points = np.random.rand(n_points, self.n_obj)
        points = points / np.sqrt(np.sum(points**2, axis=1, keepdims=True))
        return points


# ================= DC2DTLZ1.py =================

import numpy as np
from pymoo.problems import get_problem
from .Base import BaseCMOP

class DC2DTLZ1(BaseCMOP):
    """
    DC2-DTLZ1: DTLZ1 with two constraints on g(xm):
    cos(3π * g) > 0.5  and  exp(-g) > 0.5.
    """
    def __init__(self, n_obj=3):
        super().__init__(n_obj=n_obj)
        self.dtlz1 = get_problem("dtlz1", n_var=self.n_var, n_obj=self.n_obj)

    def _g(self, X):
        # DTLZ1 g function
        xm = X[:, self.n_obj-1:]
        k = xm.shape[1]
        sum1 = np.sum((xm - 0.5)**2 - np.cos(20 * np.pi * (xm - 0.5)), axis=1)
        return 100 * (k + sum1)

    def _evaluate(self, X, out, *args, **kwargs):
        f = self.dtlz1.evaluate(X)

        g = self._g(X)
        c1 = np.cos(3 * np.pi * g) - 0.5
        c2 = np.exp(-g) - 0.5
        G = np.column_stack([-c1, -c2])
        out["G"] = G
        out["F"] = f

    def get_pareto_front(self, n_points=10000):
        # DTLZ1 PF
        points = np.random.rand(n_points, self.n_obj)
        points = points / np.sum(points, axis=1, keepdims=True) * 0.5
        return points

    
# ================= DC2DTLZ3.py =================

import numpy as np
from pymoo.problems import get_problem
from .Base import BaseCMOP

class DC2DTLZ3(BaseCMOP):
    """
    DC2-DTLZ3: DTLZ3 with two constraints on g(xm):
    cos(3π * g) > 0.5  and  exp(-g) > 0.5.
    """
    def __init__(self, n_obj=3):
        super().__init__(n_obj=n_obj)
        self.dtlz3 = get_problem("dtlz3", n_var=self.n_var, n_obj=self.n_obj)

    def _g(self, X):
        # DTLZ3 g function (same as DTLZ1)
        xm = X[:, self.n_obj-1:]
        k = xm.shape[1]
        sum1 = np.sum((xm - 0.5)**2 - np.cos(20 * np.pi * (xm - 0.5)), axis=1)
        return 100 * (k + sum1)

    def _evaluate(self, X, out, *args, **kwargs):
        f = self.dtlz3.evaluate(X)

        g = self._g(X)
        c1 = np.cos(3 * np.pi * g) - 0.5
        c2 = np.exp(-g) - 0.5
        G = np.column_stack([-c1, -c2])
        out["G"] = G
        out["F"] = f

    def get_pareto_front(self, n_points=10000):
        # DTLZ3 PF (unit sphere)
        points = np.random.rand(n_points, self.n_obj)
        points = points / np.sqrt(np.sum(points**2, axis=1, keepdims=True))
        return points
   

# ================= DC3DTLZ1.py =================

import numpy as np
from pymoo.problems import get_problem
from .Base import BaseCMOP

class DC3DTLZ1(BaseCMOP):
    """
    DC3-DTLZ1: DTLZ1 with m+1 constraints:
    For j=1..m: cos(3π * x_j) > 0.5
    and cos(3π * g) > 0.5.
    """
    def __init__(self, n_obj=3):
        super().__init__(n_obj=n_obj)
        self.dtlz1 = get_problem("dtlz1", n_var=self.n_var, n_obj=self.n_obj)

    def _g(self, X):
        xm = X[:, self.n_obj-1:]
        k = xm.shape[1]
        sum1 = np.sum((xm - 0.5)**2 - np.cos(20 * np.pi * (xm - 0.5)), axis=1)
        return 100 * (k + sum1)

    def _evaluate(self, X, out, *args, **kwargs):
        f = self.dtlz1.evaluate(X)

        # Constraints on first m decision variables
        cons = []
        for j in range(min(self.n_obj, self.n_var)):
            c = np.cos(3 * np.pi * X[:, j]) - 0.5
            cons.append(-c)
        # Constraint on g
        g = self._g(X)
        c_g = np.cos(3 * np.pi * g) - 0.5
        cons.append(-c_g)
        out["G"] = np.column_stack(cons)
        out["F"] = f

    def get_pareto_front(self, n_points=10000):
        # DTLZ1 PF
        points = np.random.rand(n_points, self.n_obj)
        points = points / np.sum(points, axis=1, keepdims=True) * 0.5
        return points

    
# ================= DC3DTLZ3.py =================

import numpy as np
from pymoo.problems import get_problem
from .Base import BaseCMOP

class DC3DTLZ3(BaseCMOP):
    """
    DC3-DTLZ3: DTLZ3 with m+1 constraints:
    For j=1..m: cos(3π * x_j) > 0.5
    and cos(3π * g) > 0.5.
    """
    def __init__(self, n_obj=3):
        super().__init__(n_obj=n_obj)
        self.dtlz3 = get_problem("dtlz3", n_var=self.n_var, n_obj=self.n_obj)

    def _g(self, X):
        xm = X[:, self.n_obj-1:]
        k = xm.shape[1]
        sum1 = np.sum((xm - 0.5)**2 - np.cos(20 * np.pi * (xm - 0.5)), axis=1)
        return 100 * (k + sum1)

    def _evaluate(self, X, out, *args, **kwargs):
        f = self.dtlz3.evaluate(X)

        cons = []
        for j in range(min(self.n_obj, self.n_var)):
            c = np.cos(3 * np.pi * X[:, j]) - 0.5
            cons.append(-c)
        g = self._g(X)
        c_g = np.cos(3 * np.pi * g) - 0.5
        cons.append(-c_g)
        out["G"] = np.column_stack(cons)
        out["F"] = f

    def get_pareto_front(self, n_points=10000):
        # DTLZ3 PF (unit sphere)
        points = np.random.rand(n_points, self.n_obj)
        points = points / np.sqrt(np.sum(points**2, axis=1, keepdims=True))
        return points


# ================= test.py =================

import numpy as np
from Problems import *

problems = [
    C1DTLZ1(3), C1DTLZ3(3), C2DTLZ2(3), C3DTLZ1(3), C3DTLZ4(3),
    DC1DTLZ1(3), DC1DTLZ3(3), DC2DTLZ1(3), DC2DTLZ3(3), DC3DTLZ1(3), DC3DTLZ3(3)
]

for prob in problems:
    # Sample random solutions
    X = np.random.uniform(prob.xl, prob.xu, size=(1000, prob.n_var))
    # Evaluate
    out = {}
    prob._evaluate(X, out)
    f, G = out["F"], out["G"]
    feasible = np.all(G <= 0, axis=1)
    print(f"{prob.__class__.__name__}: feasible ratio = {np.mean(feasible)*100:.2f}%")



"""
(venv) (base) rajm012@rajm012:~/Desktop/6th Semester/3-Quant Finance (Prof. Manoj Thankur)/Project$ python -m Problems.test
C1DTLZ1: feasible ratio = 0.00%
C1DTLZ3: feasible ratio = 100.00%
C2DTLZ2: feasible ratio = 100.00%
C3DTLZ1: feasible ratio = 100.00%
C3DTLZ4: feasible ratio = 1.20%
DC1DTLZ1: feasible ratio = 33.80%
DC1DTLZ3: feasible ratio = 33.30%
DC2DTLZ1: feasible ratio = 0.00%
DC2DTLZ3: feasible ratio = 0.00%
DC3DTLZ1: feasible ratio = 1.00%
DC3DTLZ3: feasible ratio = 1.20%
(venv) (base) rajm012@rajm012:~/Desktop/6th Semester/3-Quant Finance (Prof. Manoj Thankur)/Project$ 
"""

# ================= __init__.py =================

from .C1DTLZ1 import C1DTLZ1
from .C1DTLZ3 import C1DTLZ3
from .C2DTLZ2 import C2DTLZ2
from .C3DTLZ1 import C3DTLZ1
from .C3DTLZ4 import C3DTLZ4
from .DC1DTLZ1 import DC1DTLZ1
from .DC1DTLZ3 import DC1DTLZ3
from .DC2DTLZ1 import DC2DTLZ1
from .DC2DTLZ3 import DC2DTLZ3
from .DC3DTLZ1 import DC3DTLZ1
from .DC3DTLZ3 import DC3DTLZ3
from .Base import BaseCMOP

__all__ = [
    "BaseCMOP",
    "C1DTLZ1",
    "C1DTLZ3",
    "C2DTLZ2",
    "C3DTLZ1",
    "C3DTLZ4",
    "DC1DTLZ1",
    "DC1DTLZ3",
    "DC2DTLZ1",
    "DC2DTLZ3",
    "DC3DTLZ1",
    "DC3DTLZ3",
]

