"""
Peer Comparison Algorithms
===========================
Implements the 5 constrained EMO algorithms used for comparison in the paper:

1. C-MOEA/D   — MOEA/D with feasibility-based update
2. C-NSGA-III — NSGA-III with constrained dominance relation
3. C-MOEA/DD  — Dominance+Decomposition (Li et al. 2015)
4. I-DBEA     — Indicator-based (Asafuddoula et al. 2015)
5. CMOEA      — Adaptive penalty (Woldesenbet et al. 2009)

These are simplified reference implementations matching the paper descriptions.
"""

import numpy as np
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import (
    get_weight_vectors, get_population_size,
    simulated_binary_crossover, polynomial_mutation,
    fast_non_dominated_sort, normalize_objectives,
    fast_association, tchebycheff
)
from Algorithms.ctaea import Solution, _nondominated_mask


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def constrained_dominates(s1: Solution, s2: Solution) -> bool:
    """
    Constrained dominance relation (Deb et al., NSGA-II).
    s1 constrained-dominates s2 if:
      (a) s1 feasible, s2 not, OR
      (b) both infeasible and CV(s1) < CV(s2), OR
      (c) both feasible and s1 dominates s2
    """
    if s1.feasible and not s2.feasible:
        return True
    if not s1.feasible and s2.feasible:
        return False
    if not s1.feasible and not s2.feasible:
        return s1.CV < s2.CV
    # Both feasible
    return np.all(s1.F <= s2.F) and np.any(s1.F < s2.F)


def constrained_non_dominated_sort(solutions: list) -> list:
    """
    Non-dominated sort using the constrained dominance relation.
    Returns list of fronts.
    """
    n = len(solutions)
    dom_count = np.zeros(n, dtype=int)
    dom_set = [[] for _ in range(n)]
    fronts = [[]]

    for i in range(n):
        for j in range(i + 1, n):
            if constrained_dominates(solutions[i], solutions[j]):
                dom_set[i].append(j)
                dom_count[j] += 1
            elif constrained_dominates(solutions[j], solutions[i]):
                dom_set[j].append(i)
                dom_count[i] += 1

    for i in range(n):
        if dom_count[i] == 0:
            fronts[0].append(i)

    k = 0
    while fronts[k]:
        next_front = []
        for i in fronts[k]:
            for j in dom_set[i]:
                dom_count[j] -= 1
                if dom_count[j] == 0:
                    next_front.append(j)
        k += 1
        fronts.append(next_front)

    return [f for f in fronts if f]


def _init_pop(problem, N: int):
    """Randomly initialize a population of N solutions."""
    xl, xu = problem.xl, problem.xu
    pop = []
    for _ in range(N):
        x = np.random.uniform(xl, xu)
        F, CV = problem.evaluate(x)
        pop.append(Solution(x, F, CV))
    return pop, N


def _sbx_pm_offspring(pool: list, N: int, xl, xu,
                       eta_c=30, eta_m=20, prob_c=0.9):
    """Generate N offspring from pool using SBX + PM."""
    offspring = []
    while len(offspring) < N:
        i1, i2 = np.random.choice(len(pool), 2, replace=False)
        c1, c2 = simulated_binary_crossover(
            pool[i1].x, pool[i2].x, xl, xu, eta_c, prob_c
        )
        for cx in [c1, c2]:
            if len(offspring) < N:
                cx = polynomial_mutation(cx, xl, xu, eta_m)
                offspring.append(cx)
    return offspring


# ---------------------------------------------------------------------------
# 1. C-MOEA/D
# ---------------------------------------------------------------------------

class CMOEAD:
    """
    C-MOEA/D: MOEA/D with feasibility-based update rule.
    Reference: Jain & Deb, 2014 (NSGA-III paper, Section IV)
    """
    name = 'C-MOEA/D'

    def __init__(self, problem, m, N=None, max_fe=None, seed=None, verbose=False):
        self.problem = problem
        self.m = m
        self.N = N or get_population_size(m)
        self.max_fe = max_fe
        self.verbose = verbose
        if seed is not None:
            np.random.seed(seed)

        self.W = get_weight_vectors(m, self.N)
        self.N = len(self.W)
        self.T = max(2, self.N // 10)  # neighborhood size

        # Build neighborhoods
        dists = np.array([[np.linalg.norm(self.W[i] - self.W[j])
                           for j in range(self.N)]
                          for i in range(self.N)])
        self.neighbors = np.argsort(dists, axis=1)[:, :self.T]

        self.pop = []
        self.fe_count = 0
        self.z_ideal = np.full(m, np.inf)

    def run(self):
        xl, xu = self.problem.xl, self.problem.xu

        # Initialize
        self.pop, _ = _init_pop(self.problem, self.N)
        self.fe_count = self.N
        for s in self.pop:
            self.z_ideal = np.minimum(self.z_ideal, s.F)

        while self.fe_count < self.max_fe:
            for i in range(self.N):
                # Select parents from neighborhood
                nb = self.neighbors[i]
                p1_idx, p2_idx = np.random.choice(nb, 2, replace=False)
                c_x, _ = simulated_binary_crossover(
                    self.pop[p1_idx].x, self.pop[p2_idx].x, xl, xu
                )
                c_x = polynomial_mutation(c_x, xl, xu)
                F, CV = self.problem.evaluate(c_x)
                child = Solution(c_x, F, CV)
                self.fe_count += 1
                self.z_ideal = np.minimum(self.z_ideal, F)

                # Update neighbors using feasibility rule
                for j in nb:
                    parent = self.pop[j]
                    # Feasibility rule:
                    if child.feasible and not parent.feasible:
                        self.pop[j] = child
                    elif not child.feasible and parent.feasible:
                        pass  # parent survives
                    elif not child.feasible and not parent.feasible:
                        if child.CV < parent.CV:
                            self.pop[j] = child
                    else:
                        # Both feasible: compare aggregation
                        f_n_c = (child.F - self.z_ideal) / (np.maximum(np.max(np.array([s.F for s in self.pop]), axis=0) - self.z_ideal, 1e-10))
                        f_n_p = (parent.F - self.z_ideal) / (np.maximum(np.max(np.array([s.F for s in self.pop]), axis=0) - self.z_ideal, 1e-10))
                        tch_c = tchebycheff(f_n_c, self.W[j], np.zeros(self.m))
                        tch_p = tchebycheff(f_n_p, self.W[j], np.zeros(self.m))
                        if tch_c <= tch_p:
                            self.pop[j] = child

                if self.fe_count >= self.max_fe:
                    break

        return self.pop

    def get_pareto_front(self):
        feasible = [s for s in self.pop if s.feasible]
        if not feasible:
            return np.array([]).reshape(0, self.m)
        F = np.array([s.F for s in feasible])
        nd = _nondominated_mask(F)
        return F[nd]


# ---------------------------------------------------------------------------
# 2. C-NSGA-III
# ---------------------------------------------------------------------------

class CNSGAIII:
    """
    C-NSGA-III: NSGA-III with constrained dominance relation.
    Reference: Jain & Deb, IEEE TEVC 2014
    """
    name = 'C-NSGA-III'

    def __init__(self, problem, m, N=None, max_fe=None, seed=None, verbose=False):
        self.problem = problem
        self.m = m
        self.N = N or get_population_size(m)
        self.max_fe = max_fe
        self.verbose = verbose
        if seed is not None:
            np.random.seed(seed)

        self.W = get_weight_vectors(m, self.N)
        self.N = len(self.W)
        self.pop = []
        self.fe_count = 0

    def run(self):
        xl, xu = self.problem.xl, self.problem.xu

        self.pop, _ = _init_pop(self.problem, self.N)
        self.fe_count = self.N

        while self.fe_count < self.max_fe:
            # Tournament selection + SBX + PM
            Q = []
            offspring_xs = _sbx_pm_offspring(self.pop, self.N, xl, xu)
            for cx in offspring_xs:
                F, CV = self.problem.evaluate(cx)
                Q.append(Solution(cx, F, CV))
                self.fe_count += 1
                if self.fe_count >= self.max_fe:
                    break

            R = self.pop + Q
            new_pop = self._select(R)
            self.pop = new_pop

        return self.pop

    def _select(self, R: list) -> list:
        """NSGA-III environmental selection with constrained dominance."""
        fronts = constrained_non_dominated_sort(R)

        new_pop = []
        last_front = []
        for front in fronts:
            if len(new_pop) + len(front) <= self.N:
                new_pop.extend([R[i] for i in front])
            else:
                last_front = [R[i] for i in front]
                break

        if len(new_pop) == self.N:
            return new_pop

        # Fill from last_front using reference-point niching
        needed = self.N - len(new_pop)
        if not last_front:
            return new_pop[:self.N]

        all_so_far = new_pop + last_front
        F_all = np.array([s.F for s in all_so_far])
        z_ideal = np.min(F_all, axis=0)
        z_nadir = np.max(F_all, axis=0)
        denom = z_nadir - z_ideal
        denom[denom < 1e-10] = 1e-10
        F_norm = (F_all - z_ideal) / denom

        # Associate to reference points
        sr = fast_association(F_norm, self.W)

        # Niche counting for new_pop
        niche_count = np.zeros(len(self.W), dtype=int)
        for i in range(len(new_pop)):
            niche_count[sr[i]] += 1

        selected = list(new_pop)
        remaining = list(range(len(new_pop), len(all_so_far)))

        while len(selected) < self.N and remaining:
            # Find minimum niche count
            min_nc = min(niche_count[sr[i]] for i in remaining)
            candidates = [i for i in remaining if niche_count[sr[i]] == min_nc]
            chosen = np.random.choice(candidates)
            selected.append(all_so_far[chosen])
            niche_count[sr[chosen]] += 1
            remaining.remove(chosen)

        return selected[:self.N]

    def get_pareto_front(self):
        feasible = [s for s in self.pop if s.feasible]
        if not feasible:
            return np.array([]).reshape(0, self.m)
        F = np.array([s.F for s in feasible])
        nd = _nondominated_mask(F)
        return F[nd]


# ---------------------------------------------------------------------------
# 3. C-MOEA/DD (Li et al. 2015)
# ---------------------------------------------------------------------------

class CMOEAD_DD:
    """
    C-MOEA/DD: Combines Pareto dominance and decomposition.
    Reference: Li et al., IEEE TEVC 2015
    """
    name = 'C-MOEA/DD'

    def __init__(self, problem, m, N=None, max_fe=None, seed=None, verbose=False):
        self.problem = problem
        self.m = m
        self.N = N or get_population_size(m)
        self.max_fe = max_fe
        self.verbose = verbose
        if seed is not None:
            np.random.seed(seed)

        self.W = get_weight_vectors(m, self.N)
        self.N = len(self.W)
        self.pop = []
        self.fe_count = 0
        self.z_ideal = np.full(m, np.inf)

    def run(self):
        xl, xu = self.problem.xl, self.problem.xu

        self.pop, _ = _init_pop(self.problem, self.N)
        self.fe_count = self.N
        for s in self.pop:
            self.z_ideal = np.minimum(self.z_ideal, s.F)

        while self.fe_count < self.max_fe:
            offspring_xs = _sbx_pm_offspring(self.pop, self.N, xl, xu)
            Q = []
            for cx in offspring_xs:
                F, CV = self.problem.evaluate(cx)
                Q.append(Solution(cx, F, CV))
                self.z_ideal = np.minimum(self.z_ideal, F)
                self.fe_count += 1
                if self.fe_count >= self.max_fe:
                    break

            R = self.pop + Q
            self.pop = self._select(R)

        return self.pop

    def _select(self, R: list) -> list:
        """Selection: nondom sort + subregion density trimming, infeasible allowed in isolated subregions."""
        fronts = constrained_non_dominated_sort(R)

        new_pop = []
        for front in fronts:
            if len(new_pop) + len(front) <= self.N:
                new_pop.extend([R[i] for i in front])
            else:
                needed = self.N - len(new_pop)
                front_sols = [R[i] for i in front]
                # Trim by subregion density
                all_so_far = new_pop + front_sols
                F_all = np.array([s.F for s in all_so_far])
                z_nad = np.max(F_all, axis=0)
                denom = z_nad - self.z_ideal
                denom[denom < 1e-10] = 1e-10
                F_norm = (F_all - self.z_ideal) / denom
                sr = fast_association(F_norm, self.W)

                niche_count = np.zeros(len(self.W), dtype=int)
                for i in range(len(new_pop)):
                    niche_count[sr[i]] += 1

                remaining = list(range(len(new_pop), len(all_so_far)))
                while len(new_pop) < self.N and remaining:
                    min_nc = min(niche_count[sr[i]] for i in remaining)
                    candidates = [i for i in remaining if niche_count[sr[i]] == min_nc]
                    # Among candidates, select by aggregation value
                    best = min(candidates, key=lambda i: tchebycheff(
                        F_norm[i], self.W[sr[i]], np.zeros(self.m)
                    ))
                    new_pop.append(all_so_far[best])
                    niche_count[sr[best]] += 1
                    remaining.remove(best)
                break

        return new_pop[:self.N]

    def get_pareto_front(self):
        feasible = [s for s in self.pop if s.feasible]
        if not feasible:
            return np.array([]).reshape(0, self.m)
        F = np.array([s.F for s in feasible])
        nd = _nondominated_mask(F)
        return F[nd]


# ---------------------------------------------------------------------------
# 4. I-DBEA (Asafuddoula et al. 2015)
# ---------------------------------------------------------------------------

class IDBEA:
    """
    I-DBEA: Indicator-based Decomposition-based EA.
    Uses adaptive epsilon constraint handling.
    Reference: Asafuddoula et al., IEEE TEVC 2015
    """
    name = 'I-DBEA'

    def __init__(self, problem, m, N=None, max_fe=None, seed=None, verbose=False):
        self.problem = problem
        self.m = m
        self.N = N or get_population_size(m)
        self.max_fe = max_fe
        self.verbose = verbose
        if seed is not None:
            np.random.seed(seed)

        self.W = get_weight_vectors(m, self.N)
        self.N = len(self.W)
        self.pop = []
        self.fe_count = 0
        self.z_ideal = np.full(m, np.inf)

    def run(self):
        xl, xu = self.problem.xl, self.problem.xu

        self.pop, _ = _init_pop(self.problem, self.N)
        self.fe_count = self.N
        for s in self.pop:
            self.z_ideal = np.minimum(self.z_ideal, s.F)

        # Adaptive epsilon: starts at max CV, decreases over generations
        all_cv = [s.CV for s in self.pop]
        epsilon = np.max(all_cv) if all_cv else 1.0
        gen = 0

        while self.fe_count < self.max_fe:
            offspring_xs = _sbx_pm_offspring(self.pop, self.N, xl, xu)
            Q = []
            for cx in offspring_xs:
                F, CV = self.problem.evaluate(cx)
                Q.append(Solution(cx, F, CV))
                self.z_ideal = np.minimum(self.z_ideal, F)
                self.fe_count += 1
                if self.fe_count >= self.max_fe:
                    break

            # Update with epsilon-constraint
            for child in Q:
                # Find associated subregion
                F_all = np.array([s.F for s in self.pop])
                z_nad = np.max(F_all, axis=0)
                denom = z_nad - self.z_ideal
                denom[denom < 1e-10] = 1e-10
                f_norm = (child.F - self.z_ideal) / denom
                F_norm_pop = (F_all - self.z_ideal) / denom
                sr = fast_association(f_norm.reshape(1, -1), self.W)[0]

                parent = self.pop[sr]
                child_eps_ok = child.CV <= epsilon
                parent_eps_ok = parent.CV <= epsilon

                if child_eps_ok and parent_eps_ok:
                    tc = tchebycheff(f_norm, self.W[sr], np.zeros(self.m))
                    f_p_norm = (parent.F - self.z_ideal) / denom
                    tp = tchebycheff(f_p_norm, self.W[sr], np.zeros(self.m))
                    if tc < tp:
                        self.pop[sr] = child
                elif child_eps_ok and not parent_eps_ok:
                    self.pop[sr] = child
                elif not child_eps_ok and not parent_eps_ok:
                    if child.CV < parent.CV:
                        self.pop[sr] = child

            # Decay epsilon
            gen += 1
            total_gen = self.max_fe // self.N
            epsilon = max(0.0, epsilon * (1 - gen / total_gen))

        return self.pop

    def get_pareto_front(self):
        feasible = [s for s in self.pop if s.feasible]
        if not feasible:
            return np.array([]).reshape(0, self.m)
        F = np.array([s.F for s in feasible])
        nd = _nondominated_mask(F)
        return F[nd]


# ---------------------------------------------------------------------------
# 5. CMOEA (Woldesenbet et al. 2009)
# ---------------------------------------------------------------------------

class CMOEA:
    """
    CMOEA: Constraint handling via adaptive penalty + distance measure.
    Reference: Woldesenbet et al., IEEE TEVC 2009
    Uses NSGA-II as the base optimizer on the modified objectives.
    """
    name = 'CMOEA'

    def __init__(self, problem, m, N=None, max_fe=None, seed=None, verbose=False):
        self.problem = problem
        self.m = m
        self.N = N or get_population_size(m)
        self.max_fe = max_fe
        self.verbose = verbose
        if seed is not None:
            np.random.seed(seed)

        self.pop = []
        self.fe_count = 0

    def run(self):
        xl, xu = self.problem.xl, self.problem.xu

        self.pop, _ = _init_pop(self.problem, self.N)
        self.fe_count = self.N

        while self.fe_count < self.max_fe:
            offspring_xs = _sbx_pm_offspring(self.pop, self.N, xl, xu)
            Q = []
            for cx in offspring_xs:
                F, CV = self.problem.evaluate(cx)
                Q.append(Solution(cx, F, CV))
                self.fe_count += 1
                if self.fe_count >= self.max_fe:
                    break

            R = self.pop + Q
            # Compute modified objectives
            F_mod = self._modified_objectives(R)

            fronts = fast_non_dominated_sort(F_mod)
            new_pop = []
            for front in fronts:
                if len(new_pop) + len(front) <= self.N:
                    new_pop.extend([R[i] for i in front])
                else:
                    # Crowding distance
                    rem = [R[i] for i in front]
                    F_rem = F_mod[list(front)]
                    cd = self._crowding_distance(F_rem)
                    sorted_by_cd = sorted(
                        range(len(rem)), key=lambda i: -cd[i]
                    )
                    needed = self.N - len(new_pop)
                    new_pop.extend([rem[sorted_by_cd[i]] for i in range(needed)])
                    break

            self.pop = new_pop[:self.N]

        return self.pop

    def _modified_objectives(self, pop: list) -> np.ndarray:
        """
        Transform objectives using adaptive penalty (Eq. in Woldesenbet 2009).
        f'_i(x) = f_i(x) + penalty(x)  for each objective
        """
        F_mat = np.array([s.F for s in pop])
        CV_vec = np.array([s.CV for s in pop])

        # Adaptive penalty based on average constraint violation
        avg_cv = np.mean(CV_vec[CV_vec > 0]) if np.any(CV_vec > 0) else 1.0
        avg_cv = max(avg_cv, 1e-10)

        f_range = np.max(F_mat, axis=0) - np.min(F_mat, axis=0)
        f_range[f_range < 1e-10] = 1.0

        # Penalty = sum of f_range * (CV / avg_cv) for infeasible
        penalty = np.where(
            CV_vec > 0,
            np.sum(f_range) * (CV_vec / avg_cv),
            0.0
        )

        F_modified = F_mat + penalty[:, np.newaxis]
        return F_modified

    def _crowding_distance(self, F: np.ndarray) -> np.ndarray:
        n, m = F.shape
        cd = np.zeros(n)
        for j in range(m):
            idx = np.argsort(F[:, j])
            cd[idx[0]] = cd[idx[-1]] = np.inf
            f_range = F[idx[-1], j] - F[idx[0], j]
            if f_range < 1e-10:
                continue
            for k in range(1, n - 1):
                cd[idx[k]] += (F[idx[k+1], j] - F[idx[k-1], j]) / f_range
        return cd

    def get_pareto_front(self):
        feasible = [s for s in self.pop if s.feasible]
        if not feasible:
            return np.array([]).reshape(0, self.m)
        F = np.array([s.F for s in feasible])
        nd = _nondominated_mask(F)
        return F[nd]


# ---------------------------------------------------------------------------
# Algorithm factory
# ---------------------------------------------------------------------------

ALGORITHM_CLASSES = {
    'C-TAEA':    None,  # imported separately
    'C-MOEA/D':  CMOEAD,
    'C-NSGA-III': CNSGAIII,
    'C-MOEA/DD': CMOEAD_DD,
    'I-DBEA':    IDBEA,
    'CMOEA':     CMOEA,
}
