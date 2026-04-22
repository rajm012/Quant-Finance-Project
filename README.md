
# Paper Status Quo

## Folder Structure

- Algos/            # All 6 SOTAs which we are using
- Metrics/          # Other functions 
- Others/           # Random files used
- Papers/           # Reading resources for this project
- Problems/         # 11 benchmarking problems
- venv/             # virtual env for this project
- .gitignore        # self explainatory
- README.md         # self explainatory
- requirements.txt  # self explainatory

## Files explained

### Problems
├── Base.py
├── C1DTLZ1.py
├── C1DTLZ3.py
├── C2DTLZ2.py
├── C3DTLZ1.py
├── C3DTLZ4.py
├── DC1DTlZ1.py
├── DC1DTlZ3.py
├── DC2DTlZ1.py
├── DC2DTlZ3.py
├── DC3DTlZ1.py
├── DC3DTlZ3.py
├── test.py
└── __init__.py


### Metrices
├── hv.py
├── igd.py
├── __init__.py
├── reference_points.py
├── utils.py
└── test.py

1) IGD
2) Hypervolume (HV)

#### Summary Table of Reference Points

| Problem Type | PF Type | Reference Point for HV |
|--------------|---------|------------------------|
| C1-DTLZ1 | Linear (sum=0.5) | (1.1, ..., 1.1) |
| C1-DTLZ3 | Sphere quadrant | (1.1, ..., 1.1) |
| C2-DTLZ2 | Sphere quadrant | (1.1, ..., 1.1) |
| C3-DTLZ1 | Linear (sum=0.5) | (1.1, ..., 1.1) |
| C3-DTLZ4 | Sphere quadrant | (2.1, ..., 2.1) |
| DC1-DTLZ1 | Linear | (1.1, ..., 1.1) |
| DC1-DTLZ3 | Sphere | (1.1, ..., 1.1) |
| DC2-DTLZ1 | Linear | (1.1, ..., 1.1) |
| DC2-DTLZ3 | Sphere | (1.1, ..., 1.1) |
| DC3-DTLZ1 | Linear | (1.1, ..., 1.1) |
| DC3-DTLZ3 | Sphere | (1.1, ..., 1.1) |



### Algorithms
1) C-TAEA
2) C-NSGA-III
3) C-MOEA/D
4) C-MOEA/DD
5) I-DBEA
6) CMOEA


## Path
Problems → Metrics → Algorithms → Simulations → Plots → Tables

Step 0 — Problems [Done]
Step 1 — Metrics [Done]
Step 2 — Algorithms [Working]
Step 3 — Experiment runner
Step 4 — Reproduce tables
Step 5 — Extensions / research


Problems/
    C1DTLZ1.py
    C1DTLZ3.py
    ...
    DC3DTLZ4.py

Metrics/
    igd.py
    hv.py
    reference_points.py
    test.py
    utils.py
    __init__.py

Algorithms/
Experiments/
Results/
Analysis/


