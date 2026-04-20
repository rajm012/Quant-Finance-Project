# import numpy as np
# from .C1DTLZ1 import C1DTLZ1
# from .C1DTLZ3 import C1DTLZ3
# from .C2DTLZ2 import C2DTLZ2
# from .C3DTLZ1 import C3DTLZ1
# from .C3DTLZ4 import C3DTLZ4
# from .DC1DTLZ1 import DC1DTLZ1
# from .DC1DTLZ3 import DC1DTLZ3
# from .DC2DTLZ1 import DC2DTLZ1
# from .DC2DTLZ3 import DC2DTLZ3
# from .DC3DTLZ1 import DC3DTLZ1
# from .DC3DTLZ3 import DC3DTLZ3

# problems = [
#     C1DTLZ1(3), C1DTLZ3(3), C2DTLZ2(3), C3DTLZ1(3), C3DTLZ4(3),
#     DC1DTLZ1(3), DC1DTLZ3(3), DC2DTLZ1(3), DC2DTLZ3(3), DC3DTLZ1(3), DC3DTLZ3(3)
# ]

# def TEST(problem, NSam=10000):
#     """Sample random solutions, compute feasibility ratio and objective bounds"""
    
#     X = np.random.uniform(problem.xl, problem.xu, size=(NSam, problem.n_var))
#     feasibleCnt = 0
#     allF = []
    
#     for x in X:
#         f, _, CV = problem.evaluate(x)
#         allF.append(f)
#         if CV == 0:
#             feasibleCnt += 1
#     feasibleRatio = feasibleCnt / NSam * 100
#     allF = np.array(allF)
    
#     print(f"{problem.__class__.__name__}, m={problem.n_obj}")
#     print(f"  Feasible ratio: {feasibleRatio:.2f}% (supplement Table I/II expected)")
#     print(f"  Objective range: [{allF.min():.3f}, {allF.max():.3f}]")
#     print()

# if __name__ == "__main__":
#     for prob in problems:
#         TEST(prob)
        
        
        
# ===================================================================================
# ===================================================================================
# ===================================================================================



import numpy as np
from Problems import *

problems = [
    C1DTLZ1(3), C1DTLZ3(3), C2DTLZ2(3), C3DTLZ1(3), C3DTLZ4(3),
    DC1DTLZ1(3), DC1DTLZ3(3), DC2DTLZ1(3), DC2DTLZ3(3), DC3DTLZ1(3), DC3DTLZ3(3)
]

for prob in problems:
    X = np.random.uniform(prob.xl, prob.xu, size=(pow(10,7), prob.n_var))
    out = {}
    prob._evaluate(X, out)
    f, G = out["F"], out["G"]
    feasible = np.all(G <= 0, axis=1)
    print(f"{prob.__class__.__name__}: feasible ratio = {np.mean(feasible)*100:.2f}%")



"""
(venv) (base) rajm012@rajm012:~/Desktop/6th Semester/3-Quant Finance (Prof. Manoj Thankur)/Project$ python -m Problems.test
C1DTLZ1: feasible ratio = 0.00%
C1DTLZ3: feasible ratio = 100.00%
C2DTLZ2: feasible ratio = 0.00%
C3DTLZ1: feasible ratio = 100.00%
C3DTLZ4: feasible ratio = 25.91%
DC1DTLZ1: feasible ratio = 33.32%
DC1DTLZ3: feasible ratio = 33.33%
DC2DTLZ1: feasible ratio = 0.00%
DC2DTLZ3: feasible ratio = 0.00%
DC3DTLZ1: feasible ratio = 1.23%
DC3DTLZ3: feasible ratio = 1.24%
(venv) (base) rajm012@rajm012:~/Desktop/6th Semester/3-Quant Finance (Prof. Manoj Thankur)/Project$ 
"""
        
# ===================================================================================
# ===================================================================================
# ===================================================================================


# import numpy as np
# from Problems import C1DTLZ1

# for n_obj in [3, 5, 8, 10, 15]:
#     prob = C1DTLZ1(n_obj)
#     X = np.random.uniform(0, 1, size=(10, prob.n_var))
#     out = {}
#     prob._evaluate(X, out)
#     print(f"C1DTLZ1 m={n_obj}: F shape {out['F'].shape}, G shape {out['G'].shape}")

# """
# (venv) (base) rajm012@rajm012:~/Desktop/6th Semester/3-Quant Finance (Prof. Manoj Thankur)/Project$ python -m Problems.test
# C1DTLZ1 m=3: F shape (10, 3), G shape (10, 1)
# C1DTLZ1 m=5: F shape (10, 5), G shape (10, 1)
# C1DTLZ1 m=8: F shape (10, 8), G shape (10, 1)
# C1DTLZ1 m=10: F shape (10, 10), G shape (10, 1)
# C1DTLZ1 m=15: F shape (10, 15), G shape (10, 1)
# (venv) (base) rajm012@rajm012:~/Desktop/6th Semester/3-Quant Finance (Prof. Manoj Thankur)/Project$ 
# """


# ===================================================================================
# ===================================================================================
# ===================================================================================

