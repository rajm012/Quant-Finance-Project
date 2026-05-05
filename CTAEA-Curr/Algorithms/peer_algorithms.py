"""
Peer comparison algorithms used in the paper:
- C-MOEA/D
- C-NSGA-III
- C-MOEA/DD (simplified)
- I-DBEA (simplified)
- CMOEA (simplified)

These are simplified implementations matching the paper's descriptions.
"""
import numpy as np
from .utils import (
    get_N_and_H, generate_reference_vectors_table_iv,
    normalize_objectives, tchebycheff,
    associate_to_subregions, dominates,
    fast_non_dominated_sort, non_dominated_indices,
    fast_constrained_non_dominated_sort,
    sbx_crossover, polynomial_mutation,
    sbx_crossover_batch, polynomial_mutation_batch,
)


# ─────────────────────────────────────────────────────────────────────────────
# Base class for comparison algorithms
# ─────────────────────────────────────────────────────────────────────────────

class BaseEMO:
    """Shared setup for all EMO algorithms."""

    def __init__(self, problem, N=None, max_fe=None,
                 eta_c=30.0, eta_m=20.0, pc=0.9, seed=None, verbose=False):
        self.problem = problem
        self.m = problem.n_obj
        self.n_var = problem.n_var
        self.xl = problem.xl
        self.xu = problem.xu
        self.eta_c = eta_c
        self.eta_m = eta_m
        self.pc = pc
        self.verbose = verbose
        self.fe_count = 0

        if seed is not None:
            np.random.seed(seed)

        paper_N, H = get_N_and_H(self.m)
        self.W = generate_reference_vectors_table_iv(self.m)
        self.N = len(self.W) if N is None else N

        if len(self.W) != self.N:
            idx = np.random.choice(len(self.W), self.N, replace=(len(self.W) < self.N))
            self.W = self.W[idx]

        self.z_ideal = None
        self.z_nadir = None

        if max_fe is None:
            self.max_fe = 1000 * self.N
        else:
            self.max_fe = int(max_fe)

        # Population
        self.pop_X = None
        self.pop_F = None
        self.pop_G = None
        self.pop_CV = None

    def _initialize(self):
        X = np.random.uniform(self.xl, self.xu, size=(self.N, self.n_var))
        F, G = self.problem.evaluate(X)
        CV = self.problem.constraint_violation(G)
        self.fe_count += self.N
        self._update_ideal_nadir(F)
        self.pop_X, self.pop_F, self.pop_G, self.pop_CV = X, F, G, CV

    def _update_ideal_nadir(self, F):
        if self.z_ideal is None:
            self.z_ideal = np.min(F, axis=0)
            self.z_nadir = np.max(F, axis=0)
        else:
            self.z_ideal = np.minimum(self.z_ideal, np.min(F, axis=0))
            self.z_nadir = np.maximum(self.z_nadir, np.max(F, axis=0))

    def _normalize(self, F):
        return normalize_objectives(F, self.z_ideal, self.z_nadir)

    def _make_offspring(self, parent_X, parent_F, parent_CV):
        """Binary tournament + batched SBX/mutation -> N offspring (fast path)."""
        N = self.N
        n_pairs = (N + 1) // 2
        i1 = np.empty(n_pairs, dtype=np.int32)
        i2 = np.empty(n_pairs, dtype=np.int32)
        for k in range(n_pairs):
            i1[k] = self._tournament(parent_F, parent_CV)
            i2[k] = self._tournament(parent_F, parent_CV)
        P1 = parent_X[i1]
        P2 = parent_X[i2]
        C1, C2 = sbx_crossover_batch(
            P1, P2, self.xl, self.xu, self.eta_c, self.pc)
        C1 = polynomial_mutation_batch(C1, self.xl, self.xu, self.eta_m)
        C2 = polynomial_mutation_batch(C2, self.xl, self.xu, self.eta_m)
        Q_X = np.empty((N, self.n_var))
        Q_X[0::2] = C1
        if N % 2:
            Q_X[1::2] = C2[: N // 2]
        else:
            Q_X[1::2] = C2
        Q_F, Q_G = self.problem.evaluate(Q_X)
        Q_CV = self.problem.constraint_violation(Q_G)
        self.fe_count += N
        return Q_X, Q_F, Q_G, Q_CV

    def _tournament(self, F, CV, k=2):
        candidates = np.random.choice(len(F), k, replace=False)
        best = candidates[0]
        for c in candidates[1:]:
            if self._constrained_better(F[c], CV[c], F[best], CV[best]):
                best = c
        return best

    def _constrained_better(self, f1, cv1, f2, cv2):
        """Constrained dominance: True if (f1,cv1) is better than (f2,cv2)."""
        if cv1 == 0 and cv2 > 0:
            return True
        if cv2 == 0 and cv1 > 0:
            return False
        if cv1 > 0 and cv2 > 0:
            return cv1 < cv2
        return dominates(f1, f2)

    def get_feasible(self):
        feas = self.pop_CV == 0
        return self.pop_X[feas], self.pop_F[feas], self.pop_CV[feas]

    def get_nondominated(self):
        X, F, CV = self.get_feasible()
        if len(F) == 0:
            return X, F, CV
        nd = non_dominated_indices(F)
        return X[nd], F[nd], CV[nd]


# ─────────────────────────────────────────────────────────────────────────────
# C-MOEA/D
# ─────────────────────────────────────────────────────────────────────────────

class CMOEAD(BaseEMO):
    """
    C-MOEA/D: MOEA/D with feasibility-priority constraint handling.
    From Jain & Deb (2014) and Jan & Zhang (2010).

    Update rule:
    1. If offspring feasible & parent infeasible -> replace
    2. Both infeasible -> smaller CV survives
    3. Both feasible -> smaller aggregation value survives
    """

    def __init__(self, problem, N=None, max_fe=None,
                 eta_c=30.0, eta_m=20.0, pc=0.9, seed=None, verbose=False,
                 T=20):
        super().__init__(problem, N, max_fe, eta_c, eta_m, pc, seed, verbose)
        self.T = T  # neighborhood size

    def run(self):
        self._initialize()
        diff = self.W[:, np.newaxis, :] - self.W[np.newaxis, :, :]
        W_dist_sq = np.sum(diff * diff, axis=2)
        gen = 0

        while self.fe_count < self.max_fe:
            # Build neighborhood
            F_norm = self._normalize(self.pop_F)
            assign = associate_to_subregions(
                np.maximum(F_norm, 1e-10), self.W)

            # For each subproblem
            for i in range(self.N):
                if self.fe_count >= self.max_fe:
                    break

                # Neighborhood of i (T smallest weight distances, vectorized cache)
                row = W_dist_sq[i]
                Tn = min(self.T, self.N)
                if Tn >= self.N:
                    nb = np.arange(self.N)
                else:
                    nb = np.argpartition(row, Tn - 1)[:Tn]

                # Parents from neighborhood
                p1_idx, p2_idx = np.random.choice(nb, 2, replace=False)
                c_x, _ = sbx_crossover(
                    self.pop_X[p1_idx], self.pop_X[p2_idx],
                    self.xl, self.xu, self.eta_c, self.pc)
                c_x = polynomial_mutation(c_x, self.xl, self.xu, self.eta_m)

                c_F, c_G = self.problem.evaluate(c_x.reshape(1, -1))
                c_CV = self.problem.constraint_violation(c_G)[0]
                c_f = c_F[0]
                self.fe_count += 1

                self._update_ideal_nadir(c_F)

                # Update neighbors
                for j in nb:
                    parent_cv = self.pop_CV[j]
                    parent_f = self.pop_F[j]
                    w_j = self.W[j]
                    F_norm_c = self._normalize(c_f.reshape(1, -1))[0]
                    F_norm_p = self._normalize(parent_f.reshape(1, -1))[0]

                    tch_c = np.max(np.abs(F_norm_c) / np.maximum(w_j, 1e-6))
                    tch_p = np.max(np.abs(F_norm_p) / np.maximum(w_j, 1e-6))

                    replace = False
                    if c_CV == 0 and parent_cv > 0:
                        replace = True
                    elif c_CV > 0 and parent_cv > 0:
                        replace = c_CV < parent_cv
                    elif c_CV == 0 and parent_cv == 0:
                        replace = tch_c <= tch_p

                    if replace:
                        self.pop_X[j] = c_x
                        self.pop_F[j] = c_f
                        self.pop_G[j] = c_G[0]
                        self.pop_CV[j] = c_CV

            gen += 1

        return self.pop_X, self.pop_F, self.pop_CV


# ─────────────────────────────────────────────────────────────────────────────
# C-NSGA-III
# ─────────────────────────────────────────────────────────────────────────────

class CNSGAIII(BaseEMO):
    """
    C-NSGA-III: NSGA-III with constrained dominance relation.
    From Jain & Deb (2014).
    """

    def run(self):
        self._initialize()
        gen = 0

        while self.fe_count < self.max_fe:
            Q_X, Q_F, Q_G, Q_CV = self._make_offspring(
                self.pop_X, self.pop_F, self.pop_CV)

            # Combined population
            R_X = np.vstack([self.pop_X, Q_X])
            R_F = np.vstack([self.pop_F, Q_F])
            R_G = np.vstack([self.pop_G, Q_G])
            R_CV = np.concatenate([self.pop_CV, Q_CV])

            self._update_ideal_nadir(R_F)

            # Select N using constrained non-dominated sorting + reference points
            sel = self._nsga3_select(R_X, R_F, R_G, R_CV, self.N)
            self.pop_X = R_X[sel]
            self.pop_F = R_F[sel]
            self.pop_G = R_G[sel]
            self.pop_CV = R_CV[sel]

            gen += 1

        return self.pop_X, self.pop_F, self.pop_CV

    def _constrained_dominates(self, f1, cv1, f2, cv2):
        """
        Constrained dominance: f1 c-dominates f2 if:
        1. f1 feasible, f2 infeasible
        2. both infeasible, CV(f1) < CV(f2)
        3. both feasible, f1 pareto-dominates f2
        """
        if cv1 == 0 and cv2 > 0:
            return True
        if cv1 > 0 and cv2 > 0:
            return cv1 < cv2
        if cv1 == 0 and cv2 == 0:
            return dominates(f1, f2)
        return False

    def _constrained_non_dominated_sort(self, F, CV):
        """Non-dominated sort using constrained dominance (vectorized)."""
        return fast_constrained_non_dominated_sort(F, CV)

    def _nsga3_select(self, X, F, G, CV, N):
        """NSGA-III selection using constrained fronts + reference-point niching."""
        fronts, rank = self._constrained_non_dominated_sort(F, CV)

        selected = []
        for front in fronts:
            if len(selected) + len(front) <= N:
                selected.extend(front)
            else:
                # Fill using reference-point niche preservation
                remaining = N - len(selected)
                chosen = self._niche_selection(
                    selected + front, front, F, remaining)
                selected.extend(chosen)
                break

        return selected[:N]

    def _niche_selection(self, pool, candidates, F, k):
        """Simplified niche selection using association to weight vectors."""
        F_norm = self._normalize(F[pool])
        assign = associate_to_subregions(np.maximum(F_norm, 1e-10), self.W)

        pool_assign = {pool[i]: assign[i] for i in range(len(pool))}
        cand_set = set(candidates)

        counts = np.zeros(self.N, dtype=int)
        for idx in pool:
            if idx not in cand_set and pool_assign[idx] < self.N:
                counts[pool_assign[idx]] += 1

        chosen = []
        candidates_list = list(candidates)
        np.random.shuffle(candidates_list)

        for _ in range(k):
            if not candidates_list:
                break
            # Find subregion with minimum count
            cand_regions = [pool_assign.get(c, 0) for c in candidates_list]
            min_count = min(counts[r] for r in cand_regions if r < self.N)
            min_cands = [
                c for c, r in zip(candidates_list, cand_regions)
                if r < self.N and counts[r] == min_count
            ]
            # Pick random among ties
            pick = min_cands[np.random.randint(len(min_cands))]
            chosen.append(pick)
            candidates_list.remove(pick)
            r = pool_assign.get(pick, 0)
            if r < self.N:
                counts[r] += 1

        return chosen


# ─────────────────────────────────────────────────────────────────────────────
# C-MOEA/DD (simplified)
# ─────────────────────────────────────────────────────────────────────────────

class CMOEAD_DD(BaseEMO):
    """
    C-MOEA/DD: Combines Pareto dominance and decomposition.
    Li et al. (2015) with constraint handling.
    
    Simplified: uses non-dominated sorting + density trimming with
    infeasible solution preservation in isolated regions.
    """

    def run(self):
        self._initialize()
        gen = 0

        while self.fe_count < self.max_fe:
            Q_X, Q_F, Q_G, Q_CV = self._make_offspring(
                self.pop_X, self.pop_F, self.pop_CV)

            R_X = np.vstack([self.pop_X, Q_X])
            R_F = np.vstack([self.pop_F, Q_F])
            R_G = np.vstack([self.pop_G, Q_G])
            R_CV = np.concatenate([self.pop_CV, Q_CV])

            self._update_ideal_nadir(R_F)
            sel = self._select(R_X, R_F, R_G, R_CV, self.N)
            self.pop_X = R_X[sel]
            self.pop_F = R_F[sel]
            self.pop_G = R_G[sel]
            self.pop_CV = R_CV[sel]

            gen += 1

        return self.pop_X, self.pop_F, self.pop_CV

    def _select(self, X, F, G, CV, N):
        """
        1. Non-dominated sort (constrained).
        2. Fill fronts until exceeds N.
        3. In critical front: trim by subregion density + agg function.
        4. Infeasible solutions in isolated (empty) subregions are preserved.
        """
        n = len(F)
        # Constrained non-dominated sort
        fronts = self._c_nds(F, CV)

        selected = []
        for front in fronts:
            if len(selected) + len(front) <= N:
                selected.extend(front)
            else:
                remaining = N - len(selected)
                chosen = self._density_select(
                    selected + front, front, F, CV, remaining)
                selected.extend(chosen)
                break

        return selected[:N]

    def _c_nds(self, F, CV):
        """Constrained non-dominated sort (vectorized)."""
        fronts, _ = fast_constrained_non_dominated_sort(F, CV)
        return [f for f in fronts if f]

    def _c_dom(self, f1, cv1, f2, cv2):
        if cv1 == 0 and cv2 > 0:
            return True
        if cv1 > 0 and cv2 > 0:
            return cv1 < cv2
        if cv1 == 0 and cv2 == 0:
            return dominates(f1, f2)
        return False

    def _density_select(self, pool, candidates, F, CV, k):
        """Select k from candidates using subregion density."""
        F_norm = self._normalize(F[pool])
        assign = associate_to_subregions(np.maximum(F_norm, 1e-10), self.W)
        pool_assign = {pool[i]: assign[i] for i in range(len(pool))}

        cand_set = set(candidates)
        counts = np.zeros(self.N, dtype=int)
        for idx in pool:
            if idx not in cand_set and pool_assign.get(idx, 0) < self.N:
                counts[pool_assign[idx]] += 1

        chosen = []
        candidates_left = list(candidates)

        for _ in range(k):
            if not candidates_left:
                break
            # Prefer candidate in least-populated region
            cand_regions = [pool_assign.get(c, 0) for c in candidates_left]
            valid = [(c, r) for c, r in zip(candidates_left, cand_regions) if r < self.N]
            if not valid:
                valid = list(zip(candidates_left, cand_regions))

            min_cnt = min(counts[r] if r < self.N else 0 for _, r in valid)
            min_cands = [(c, r) for c, r in valid
                         if (counts[r] if r < self.N else 0) == min_cnt]

            # Among ties: pick best agg value
            best_c = None
            best_tch = np.inf
            for c, r in min_cands:
                if r < self.N:
                    w = self.W[r]
                else:
                    w = np.ones(self.m) / self.m
                fn = self._normalize(F[c:c+1])[0]
                t = np.max(np.abs(fn) / np.maximum(w, 1e-6))
                if t < best_tch:
                    best_tch = t
                    best_c = c

            if best_c is None:
                best_c = candidates_left[0]

            chosen.append(best_c)
            candidates_left.remove(best_c)
            r = pool_assign.get(best_c, 0)
            if r < self.N:
                counts[r] += 1

        return chosen


# ─────────────────────────────────────────────────────────────────────────────
# I-DBEA (simplified)
# ─────────────────────────────────────────────────────────────────────────────

class IDBEA(BaseEMO):
    """
    I-DBEA: Indicator-based Decomposition-Based EA with adaptive epsilon.
    Asafuddoula et al. (2015) with constraint handling.
    
    Simplified implementation using adaptive epsilon constraint handling.
    """

    def __init__(self, problem, N=None, max_fe=None,
                 eta_c=30.0, eta_m=20.0, pc=0.9, seed=None, verbose=False):
        super().__init__(problem, N, max_fe, eta_c, eta_m, pc, seed, verbose)
        self.epsilon0 = None   # Initial epsilon
        self.epsilon = None    # Current epsilon
        self.Tc = None         # Generation threshold

    def run(self):
        self._initialize()
        # epsilon schedule: decrease from max CV to 0 over Tc generations
        if self.pop_CV.max() > 0:
            self.epsilon = self.pop_CV.max()
        else:
            self.epsilon = 0.0

        total_gen = self.max_fe // self.N
        self.Tc = int(0.8 * total_gen)
        self.epsilon0 = self.epsilon
        gen = 0

        while self.fe_count < self.max_fe:
            # Decay epsilon
            if gen < self.Tc and self.epsilon0 > 0:
                self.epsilon = self.epsilon0 * (1 - gen / self.Tc) ** 2
            else:
                self.epsilon = 0.0

            Q_X, Q_F, Q_G, Q_CV = self._make_offspring(
                self.pop_X, self.pop_F, self.pop_CV)

            R_X = np.vstack([self.pop_X, Q_X])
            R_F = np.vstack([self.pop_F, Q_F])
            R_G = np.vstack([self.pop_G, Q_G])
            R_CV = np.concatenate([self.pop_CV, Q_CV])

            self._update_ideal_nadir(R_F)

            sel = self._select_idbea(R_X, R_F, R_G, R_CV, self.N)
            self.pop_X = R_X[sel]
            self.pop_F = R_F[sel]
            self.pop_G = R_G[sel]
            self.pop_CV = R_CV[sel]

            gen += 1

        return self.pop_X, self.pop_F, self.pop_CV

    def _eps_dominates(self, f1, cv1, f2, cv2):
        """Epsilon-constrained dominance."""
        eps = self.epsilon
        if cv1 <= eps and cv2 <= eps:
            return dominates(f1, f2)
        if cv1 <= eps and cv2 > eps:
            return True
        if cv1 > eps and cv2 > eps:
            return cv1 < cv2
        return False

    def _select_idbea(self, X, F, G, CV, N):
        """Decomposition-based selection with epsilon constraint handling."""
        n = len(F)
        F_norm = self._normalize(F)
        assign = associate_to_subregions(np.maximum(F_norm, 1e-10), self.W)

        # For each subregion, keep best solution
        subregion_best = {}
        for k in range(n):
            r = assign[k]
            if r >= self.N:
                continue
            if r not in subregion_best:
                subregion_best[r] = k
            else:
                j = subregion_best[r]
                if self._eps_dominates(F[k], CV[k], F[j], CV[j]):
                    subregion_best[r] = k
                elif not self._eps_dominates(F[j], CV[j], F[k], CV[k]):
                    # Tie: pick by agg value
                    w = self.W[r]
                    t1 = np.max(np.abs(F_norm[k]) / np.maximum(w, 1e-6))
                    t2 = np.max(np.abs(F_norm[j]) / np.maximum(w, 1e-6))
                    if t1 < t2:
                        subregion_best[r] = k

        selected = list(subregion_best.values())
        # Fill remaining
        if len(selected) < N:
            not_sel = [i for i in range(n) if i not in set(selected)]
            not_sel.sort(key=lambda i: CV[i])
            selected.extend(not_sel[:N - len(selected)])
        return selected[:N]


# ─────────────────────────────────────────────────────────────────────────────
# CMOEA (simplified)
# ─────────────────────────────────────────────────────────────────────────────

class CMOEA(BaseEMO):
    """
    CMOEA: Adaptive penalty function + NSGA-II.
    Woldesenbet et al. (2009).

    The original objective functions are transformed by adding penalty terms
    based on CV, then NSGA-II is applied to the transformed problem.
    """

    def run(self):
        self._initialize()
        gen = 0

        while self.fe_count < self.max_fe:
            Q_X, Q_F, Q_G, Q_CV = self._make_offspring(
                self.pop_X, self.pop_F, self.pop_CV)

            R_X = np.vstack([self.pop_X, Q_X])
            R_F = np.vstack([self.pop_F, Q_F])
            R_G = np.vstack([self.pop_G, Q_G])
            R_CV = np.concatenate([self.pop_CV, Q_CV])

            self._update_ideal_nadir(R_F)

            # Transform objectives with adaptive penalty
            F_pen = self._penalize(R_F, R_CV)

            # NSGA-II selection on penalized objectives
            fronts, rank = fast_non_dominated_sort(F_pen)
            sel = self._nsga2_select(fronts, F_pen, self.N)

            self.pop_X = R_X[sel]
            self.pop_F = R_F[sel]
            self.pop_G = R_G[sel]
            self.pop_CV = R_CV[sel]

            gen += 1

        return self.pop_X, self.pop_F, self.pop_CV

    def _penalize(self, F, CV):
        """
        Adaptive penalty: F_pen_i = f_i + penalty_i
        penalty_i = (d_i / (d_i + cv)) * cv  approximately
        where d_i is the distance measure for objective i.
        """
        F_norm = self._normalize(F)
        # Adaptive penalty coefficient per solution
        d = np.linalg.norm(F_norm, axis=1, keepdims=True)  # distance from ideal
        cv = CV[:, np.newaxis]
        mask = (cv > 0)
        coef = np.where(mask, d / (d + cv + 1e-10), 0.0)
        return F_norm + coef * cv

    def _nsga2_select(self, fronts, F, N):
        selected = []
        for front in fronts:
            if len(selected) + len(front) <= N:
                selected.extend(front)
            else:
                remaining = N - len(selected)
                # Crowding distance sort
                cd = self._crowding_distance(front, F)
                order = np.argsort(-cd)
                selected.extend([front[o] for o in order[:remaining]])
                break
        return selected[:N]

    def _crowding_distance(self, front, F):
        n = len(front)
        if n <= 2:
            return np.full(n, np.inf)
        cd = np.zeros(n)
        F_front = F[front]
        for obj in range(F_front.shape[1]):
            order = np.argsort(F_front[:, obj])
            cd[order[0]] = np.inf
            cd[order[-1]] = np.inf
            f_range = F_front[order[-1], obj] - F_front[order[0], obj]
            if f_range < 1e-10:
                continue
            for i in range(1, n - 1):
                cd[order[i]] += (F_front[order[i+1], obj] - F_front[order[i-1], obj]) / f_range
        return cd
