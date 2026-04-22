"""
C-TAEA: Two-Archive Evolutionary Algorithm for Constrained Multi-Objective Optimization
========================================================================================
Li et al., IEEE Transactions on Evolutionary Computation, Vol. 23, No. 2, April 2019.

Implements:
  - Algorithm 1: Association Procedure
  - Algorithm 2: Update Mechanism of CA
  - Algorithm 3: Update Mechanism of DA
  - Algorithm 4: Restricted Mating Selection
  - Algorithm 5: Tournament Selection

Reproduction: SBX crossover + polynomial mutation (Table III parameters)
"""

import numpy as np
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import (
    fast_association, tchebycheff, normalize_objectives,
    simulated_binary_crossover, polynomial_mutation,
    fast_non_dominated_sort, get_weight_vectors, get_population_size
)


# ---------------------------------------------------------------------------
# Solution representation
# ---------------------------------------------------------------------------

class Solution:
    __slots__ = ['x', 'F', 'CV', 'feasible']

    def __init__(self, x, F, CV):
        self.x = np.array(x)
        self.F = np.array(F)
        self.CV = float(CV)
        self.feasible = (self.CV == 0.0)

    def __repr__(self):
        return f"Solution(CV={self.CV:.4f}, F={self.F})"


# ---------------------------------------------------------------------------
# Algorithm 1: Association Procedure
# ---------------------------------------------------------------------------

def association_procedure(solutions: list, W: np.ndarray,
                           z_ideal: np.ndarray,
                           z_nadir: np.ndarray) -> dict:
    """
    Algorithm 1 from paper.
    Associate each solution to a subregion (weight vector).
    Returns dict: subregion_idx -> list of solutions
    """
    N = len(W)
    subregions = {i: [] for i in range(N)}

    if not solutions:
        return subregions

    F_mat = np.array([s.F for s in solutions])
    F_norm = normalize_objectives(F_mat, z_ideal, z_nadir)

    indices = fast_association(F_norm, W)
    for sol, idx in zip(solutions, indices):
        subregions[int(idx)].append(sol)

    return subregions


# ---------------------------------------------------------------------------
# Algorithm 2: Update Mechanism of CA
# ---------------------------------------------------------------------------

def update_CA(CA: list, Q: list, W: np.ndarray,
               z_ideal: np.ndarray, z_nadir: np.ndarray,
               N: int) -> list:
    """
    Algorithm 2 from paper.
    Update the Convergence-oriented Archive (CA).

    Parameters
    ----------
    CA      : current CA (list of Solution)
    Q       : offspring population (list of Solution)
    W       : weight vectors (N, m)
    z_ideal : ideal point
    z_nadir : nadir point
    N       : archive size

    Returns
    -------
    new_CA : updated CA (list of Solution, size N)
    """
    Hc = CA + Q  # hybrid population

    # Step 1: Collect feasible solutions
    Sc = [s for s in Hc if s.feasible]

    if len(Sc) == N:
        return Sc

    elif len(Sc) > N:
        # Use nondominated sorting to get best N feasible solutions
        F_mat = np.array([s.F for s in Sc])
        fronts = fast_non_dominated_sort(F_mat)

        S = []
        last_front_idx = 0
        for fi, front in enumerate(fronts):
            if len(S) + len(front) <= N:
                S.extend([Sc[j] for j in front])
                last_front_idx = fi
            else:
                # Need to trim from this front
                S.extend([Sc[j] for j in front])
                last_front_idx = fi
                break

        # If S is exactly N, done
        if len(S) == N:
            return S

        # Trim using subregion density + tchebycheff (Alg 2, lines 11-21)
        S = _trim_to_N(S, W, N)
        return S[:N]

    else:
        # |Sc| < N: use infeasible solutions to fill
        SI = [s for s in Hc if not s.feasible]

        # Sort infeasible solutions by bi-objective: (CV, tchebycheff)
        # Form combined objectives per Eq. 12
        if not SI:
            return Sc[:N] if len(Sc) >= N else Sc

        F_SI = np.array([s.F for s in SI])
        CV_SI = np.array([s.CV for s in SI])

        # Compute tchebycheff for each infeasible solution
        # Use current z_ideal and normalize
        all_F = np.array([s.F for s in Hc])
        z_id = np.min(all_F, axis=0)
        z_nad = np.max(all_F, axis=0)
        denom = z_nad - z_id
        denom[denom < 1e-10] = 1e-10

        F_norm_SI = (F_SI - z_id) / denom

        # Assign each infeasible solution to a weight vector
        sr_SI = fast_association(F_norm_SI, W)

        tch_vals = np.array([
            tchebycheff(F_norm_SI[i], W[sr_SI[i]], np.zeros(len(z_id)))
            for i in range(len(SI))
        ])

        # Bi-objective: F' = (CV, tch)
        F_bi = np.column_stack([CV_SI, tch_vals])

        # Non-dominated sort on bi-objective
        fronts = fast_non_dominated_sort(F_bi)

        S = list(Sc)
        i = 0
        while len(S) < N and i < len(fronts):
            front_sols = [SI[j] for j in fronts[i]]
            remaining = N - len(S)
            if len(front_sols) <= remaining:
                S.extend(front_sols)
            else:
                # Sort by CV ascending and trim (lower CV preferred)
                front_sols.sort(key=lambda s: s.CV)
                S.extend(front_sols[:remaining])
            i += 1

        return S[:N]


# ---------------------------------------------------------------------------
# Algorithm 3: Update Mechanism of DA
# ---------------------------------------------------------------------------

def update_DA(CA: list, DA: list, Q: list, W: np.ndarray,
               z_ideal: np.ndarray, z_nadir: np.ndarray,
               N: int) -> list:
    """
    Algorithm 3 from paper.
    Update the Diversity-oriented Archive (DA).
    Does NOT consider feasibility — explores infeasible regions too.

    Parameters
    ----------
    CA, DA : current archives
    Q      : offspring population
    W      : weight vectors (N, m)
    N      : archive size

    Returns
    -------
    new_DA : updated DA (size N)
    """
    Hd = DA + Q  # hybrid population for DA

    # Normalize all solutions (including infeasible)
    all_F = np.array([s.F for s in (Hd + CA)])
    z_id  = np.min(all_F, axis=0)
    z_nad = np.max(all_F, axis=0)
    denom = z_nad - z_id
    denom[denom < 1e-10] = 1e-10

    # Associate Hd solutions to subregions
    F_Hd  = np.array([s.F for s in Hd])
    F_Hd_norm = (F_Hd - z_id) / denom
    sr_Hd = fast_association(F_Hd_norm, W)

    # Build subregion -> [solutions in Hd] mapping
    Hd_subregions = {i: [] for i in range(N)}
    for idx, sr in enumerate(sr_Hd):
        Hd_subregions[int(sr)].append(idx)

    # Associate CA solutions to subregions
    if CA:
        F_CA = np.array([s.F for s in CA])
        F_CA_norm = (F_CA - z_id) / denom
        sr_CA = fast_association(F_CA_norm, W)
        CA_subregions = {i: [] for i in range(N)}
        for idx, sr in enumerate(sr_CA):
            CA_subregions[int(sr)].append(idx)
    else:
        CA_subregions = {i: [] for i in range(N)}

    # Iteratively fill DA
    S = []
    itr = 1
    while len(S) < N:
        for i in range(N):
            ca_count = len(CA_subregions[i])
            slots = itr - ca_count
            filled_in_this_round = 0
            for _ in range(max(0, slots)):
                if not Hd_subregions[i]:
                    break
                # Get non-dominated solutions in Hd subregion i
                hd_sol_indices = Hd_subregions[i]
                hd_sols = [Hd[j] for j in hd_sol_indices]

                # Find non-dominated solutions among them
                if len(hd_sols) == 1:
                    best_sol = hd_sols[0]
                    best_idx = hd_sol_indices[0]
                else:
                    F_local = np.array([s.F for s in hd_sols])
                    nd_mask = _nondominated_mask(F_local)
                    nd_sols = [hd_sols[k] for k in range(len(hd_sols)) if nd_mask[k]]
                    nd_idxs = [hd_sol_indices[k] for k in range(len(hd_sol_indices)) if nd_mask[k]]

                    if not nd_sols:
                        nd_sols = hd_sols
                        nd_idxs = hd_sol_indices

                    # Select best by tchebycheff with weight vector w_i
                    w = W[i]
                    tch_vals = []
                    for sol in nd_sols:
                        f_n = (sol.F - z_id) / denom
                        tch_vals.append(tchebycheff(f_n, w, np.zeros(len(w))))

                    best_local = np.argmin(tch_vals)
                    best_sol = nd_sols[best_local]
                    best_idx = nd_idxs[best_local]

                S.append(best_sol)
                Hd_subregions[i].remove(best_idx)
                filled_in_this_round += 1

                if len(S) >= N:
                    break
            if len(S) >= N:
                break

        itr += 1
        # Safety break to avoid infinite loop
        if itr > N * 2:
            # Fill remaining from Hd (any order)
            remaining = [Hd[j] for sr_list in Hd_subregions.values()
                         for j in sr_list]
            S.extend(remaining[:N - len(S)])
            break

    return S[:N]


def _nondominated_mask(F: np.ndarray) -> np.ndarray:
    """Return boolean mask: True if solution i is non-dominated in F. Vectorized."""
    n = len(F)
    if n == 0:
        return np.array([], dtype=bool)
    # dom[i,j] = True if F[j] dominates F[i]
    F_i = F[:, np.newaxis, :]   # (n,1,m)
    F_j = F[np.newaxis, :, :]   # (1,n,m)
    dominated_by = np.all(F_j <= F_i, axis=2) & np.any(F_j < F_i, axis=2)  # (n,n)
    np.fill_diagonal(dominated_by, False)
    return ~dominated_by.any(axis=1)


def _trim_to_N(S: list, W: np.ndarray, N: int) -> list:
    """
    Trim S down to N solutions using subregion-density + tchebycheff.
    Implements lines 11-21 of Algorithm 2.
    Vectorized to avoid repeated re-association.
    """
    if len(S) <= N:
        return S

    # Keep a working array and mask instead of popping
    F_arr = np.array([s.F for s in S])
    active = np.ones(len(S), dtype=bool)

    while active.sum() > N:
        F_active = F_arr[active]
        idx_active = np.where(active)[0]

        z_id  = F_active.min(axis=0)
        z_nad = F_active.max(axis=0)
        denom = np.where(z_nad - z_id < 1e-10, 1e-10, z_nad - z_id)
        F_norm = (F_active - z_id) / denom

        sr = fast_association(F_norm, W)

        # Find most crowded subregion
        counts = np.bincount(sr, minlength=len(W))
        crowded = int(np.argmax(counts))
        in_crowded = np.where(sr == crowded)[0]  # local indices in F_active

        if len(in_crowded) == 1:
            worst_local = in_crowded[0]
        else:
            # Nearest-neighbour distances within subregion
            F_crowd = F_norm[in_crowded]
            # pairwise distances
            diff = F_crowd[:, np.newaxis, :] - F_crowd[np.newaxis, :, :]
            pdist = np.linalg.norm(diff, axis=2)
            np.fill_diagonal(pdist, np.inf)
            nn_dist = pdist.min(axis=1)

            min_d = nn_dist.min()
            ties  = np.where(nn_dist == min_d)[0]
            # Break ties by worst tchebycheff
            w = W[crowded]
            z0 = np.zeros(W.shape[1])
            tch_ties = np.array([
                tchebycheff(F_crowd[k], w, z0) for k in ties
            ])
            worst_local = in_crowded[ties[np.argmax(tch_ties)]]

        # Mark solution for removal
        active[idx_active[worst_local]] = False

    return [S[i] for i in range(len(S)) if active[i]]



    """Return boolean mask: True if solution i is non-dominated in F."""
    n = len(F)
    dominated = np.zeros(n, dtype=bool)
    for i in range(n):
        for j in range(n):
            if i != j and not dominated[j]:
                if np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                    dominated[i] = True
                    break
    return ~dominated


# ---------------------------------------------------------------------------
# Algorithm 4: Restricted Mating Selection
# ---------------------------------------------------------------------------

def restricted_mating_selection(CA: list, DA: list) -> tuple:
    """
    Algorithm 4 from paper.
    Select two mating parents from CA and DA based on their
    relative non-dominated proportions.
    """
    n_CA = len(CA)
    n_DA = len(DA)
    F_Hm = np.empty((n_CA + n_DA, CA[0].F.shape[0]))
    for i, s in enumerate(CA):
        F_Hm[i] = s.F
    for i, s in enumerate(DA):
        F_Hm[n_CA + i] = s.F

    nd_mask = _nondominated_mask(F_Hm)

    rho_c = nd_mask[:n_CA].sum() / n_CA if n_CA > 0 else 0.0
    rho_d = nd_mask[n_CA:].sum() / n_DA if n_DA > 0 else 0.0

    p1 = tournament_selection(CA) if rho_c > rho_d else tournament_selection(DA)
    p2 = tournament_selection(CA) if np.random.random() < rho_c else tournament_selection(DA)

    return p1, p2


# ---------------------------------------------------------------------------
# Algorithm 5: Tournament Selection
# ---------------------------------------------------------------------------

def tournament_selection(pool: list) -> 'Solution':
    """
    Algorithm 5 from paper.
    Binary tournament selection (feasibility-driven).
    """
    idx1, idx2 = np.random.choice(len(pool), 2, replace=False)
    x1, x2 = pool[idx1], pool[idx2]

    if x1.feasible and x2.feasible:
        # Select by Pareto dominance
        if np.all(x1.F <= x2.F) and np.any(x1.F < x2.F):
            return x1
        elif np.all(x2.F <= x1.F) and np.any(x2.F < x1.F):
            return x2
        else:
            return np.random.choice([x1, x2])
    elif x1.feasible:
        return x1
    elif x2.feasible:
        return x2
    else:
        return np.random.choice([x1, x2])


# ---------------------------------------------------------------------------
# Main C-TAEA Algorithm
# ---------------------------------------------------------------------------

class CTAEA:
    """
    C-TAEA: Two-Archive Evolutionary Algorithm for CMOPs.

    Parameters
    ----------
    problem          : problem instance (has .evaluate(), .n_var, .n_obj, .xl, .xu)
    m                : number of objectives
    N                : archive size (= number of weight vectors)
    max_fe           : maximum function evaluations
    eta_c            : SBX distribution index (default 30)
    eta_m            : polynomial mutation index (default 20)
    prob_c           : SBX crossover probability (default 0.9)
    seed             : random seed
    verbose          : print progress
    """

    def __init__(self, problem, m: int = 3, N: int = None,
                 max_fe: int = None,
                 eta_c: float = 30.0, eta_m: float = 20.0,
                 prob_c: float = 0.9,
                 seed: int = None, verbose: bool = False):

        self.problem = problem
        self.m = m
        self.N = N if N is not None else get_population_size(m)
        self.max_fe = max_fe
        self.eta_c = eta_c
        self.eta_m = eta_m
        self.prob_c = prob_c
        self.verbose = verbose

        if seed is not None:
            np.random.seed(seed)

        # Generate weight vectors
        self.W = get_weight_vectors(m, self.N)
        self.N = len(self.W)  # actual size (may differ slightly)

        # Reference points
        self.z_ideal = np.full(m, np.inf)
        self.z_nadir = np.full(m, -np.inf)

        # Archives
        self.CA = []
        self.DA = []
        self.fe_count = 0

        # History
        self.history = []

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------

    def _initialize(self):
        """Randomly initialize population of size 2N (for CA and DA)."""
        xl, xu = self.problem.xl, self.problem.xu
        pop = []
        for _ in range(self.N):
            x = np.random.uniform(xl, xu)
            F, CV = self.problem.evaluate(x)
            sol = Solution(x, F, CV)
            self._update_reference_points(sol.F)
            pop.append(sol)
            self.fe_count += 1

        # Initialize both CA and DA with the same population
        self.CA = pop[:self.N]
        self.DA = list(pop)  # DA starts as copy

    def _update_reference_points(self, F: np.ndarray):
        """Update ideal and nadir points."""
        self.z_ideal = np.minimum(self.z_ideal, F)
        self.z_nadir = np.maximum(self.z_nadir, F)

    # ------------------------------------------------------------------
    # Offspring Generation
    # ------------------------------------------------------------------

    def _generate_offspring(self) -> list:
        """
        Generate N offspring using restricted mating selection + SBX + PM.
        Compute rho_c/rho_d once per generation (not per offspring pair).
        """
        xl, xu = self.problem.xl, self.problem.xu
        offspring = []

        # Compute rho once for the whole generation (Alg 4)
        n_CA, n_DA = len(self.CA), len(self.DA)
        m_obj = self.CA[0].F.shape[0]
        F_Hm = np.empty((n_CA + n_DA, m_obj))
        for i, s in enumerate(self.CA): F_Hm[i]        = s.F
        for i, s in enumerate(self.DA): F_Hm[n_CA + i] = s.F
        nd = _nondominated_mask(F_Hm)
        rho_c = float(nd[:n_CA].sum()) / n_CA if n_CA > 0 else 0.0
        rho_d = float(nd[n_CA:].sum()) / n_DA if n_DA > 0 else 0.0

        while len(offspring) < self.N:
            # p1
            p1 = tournament_selection(self.CA) if rho_c > rho_d \
                 else tournament_selection(self.DA)
            # p2
            p2 = tournament_selection(self.CA) if np.random.random() < rho_c \
                 else tournament_selection(self.DA)

            c1, c2 = simulated_binary_crossover(
                p1.x, p2.x, xl, xu, self.eta_c, self.prob_c
            )
            c1 = polynomial_mutation(c1, xl, xu, self.eta_m)
            c2 = polynomial_mutation(c2, xl, xu, self.eta_m)

            for cx in [c1, c2]:
                if len(offspring) < self.N:
                    F, CV = self.problem.evaluate(cx)
                    sol = Solution(cx, F, CV)
                    self._update_reference_points(sol.F)
                    offspring.append(sol)
                    self.fe_count += 1

        return offspring

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def run(self):
        """
        Execute C-TAEA.
        Returns (CA_final, DA_final) — the final archives.
        """
        if self.max_fe is None:
            from utils import get_fe_budget
            self.max_fe = get_fe_budget(
                getattr(self.problem, 'name', 'C1-DTLZ1'), self.m
            )

        self._initialize()

        gen = 0
        while self.fe_count < self.max_fe:
            # Generate offspring
            Q = self._generate_offspring()

            # Update ideal/nadir with entire combined population
            all_F = np.vstack([
                [s.F for s in self.CA],
                [s.F for s in self.DA],
                [s.F for s in Q]
            ])
            self.z_ideal = np.min(all_F, axis=0)
            self.z_nadir = np.max(all_F, axis=0)

            # Update CA (Algorithm 2)
            self.CA = update_CA(
                self.CA, Q, self.W,
                self.z_ideal, self.z_nadir, self.N
            )

            # Update DA (Algorithm 3)
            self.DA = update_DA(
                self.CA, self.DA, Q, self.W,
                self.z_ideal, self.z_nadir, self.N
            )

            gen += 1
            if self.verbose and gen % 50 == 0:
                n_feasible = sum(1 for s in self.CA if s.feasible)
                print(f"  Gen {gen:4d} | FE: {self.fe_count:6d}/{self.max_fe} "
                      f"| CA feasible: {n_feasible}/{self.N}")

        return self.CA, self.DA

    # ------------------------------------------------------------------
    # Result extraction
    # ------------------------------------------------------------------

    def get_pareto_front(self, archive: str = 'CA') -> np.ndarray:
        """
        Return feasible non-dominated objective vectors from specified archive.
        """
        if archive == 'CA':
            sols = self.CA
        elif archive == 'DA':
            sols = self.DA
        else:
            sols = self.CA + self.DA

        feasible = [s for s in sols if s.feasible]
        if not feasible:
            return np.array([]).reshape(0, self.m)

        F = np.array([s.F for s in feasible])
        nd_mask = _nondominated_mask(F)
        return F[nd_mask]

    def get_all_feasible(self) -> np.ndarray:
        """Return all feasible solutions from CA."""
        feasible = [s for s in self.CA if s.feasible]
        if not feasible:
            return np.array([]).reshape(0, self.m)
        return np.array([s.F for s in feasible])
