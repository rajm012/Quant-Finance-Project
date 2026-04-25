

"""
C-TAEA: Two-Archive Evolutionary Algorithm for Constrained Multiobjective Optimization.
Algorithm flow:
1. Initialize CA and DA (each size N)
2. For each generation:
   a. Restricted mating selection -> offspring Q
   b. Update CA using Algorithm 2
   c. Update DA using Algorithm 3
3. Return CA as final result
"""


import numpy as np
from .utils import (get_N_and_H, generate_reference_vectors_table_iv,
    normalize_objectives, tchebycheff, associate_to_subregions,
    dominates, fast_non_dominated_sort, non_dominated_indices,
    sbx_crossover, polynomial_mutation)


class CTAEA:
    """
    C-TAEA: Two-Archive Evolutionary Algorithm for CMOPs.

    Parameters
    ----------
    problem : BaseProblem
        The constrained multiobjective problem to solve.
    N : int or None
        Population (archive) size. If None, determined from Table IV.
    max_fe : int or None
        Maximum function evaluations. If None, uses paper defaults.
    eta_c : float
        SBX crossover distribution index (default 30).
    eta_m : float
        Polynomial mutation distribution index (default 20).
    pc : float
        Crossover probability (default 0.9).
    seed : int
        Random seed.
    verbose : bool
        Print progress every 50 generations.
    """

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

        if seed is not None:
            np.random.seed(seed)

        # Set up weight vectors
        paper_N, H = get_N_and_H(self.m)
        self.W = generate_reference_vectors_table_iv(self.m)
        self.N = len(self.W) if N is None else N

        # Adjust W to match N if needed
        if len(self.W) != self.N:
            # Resample to match N
            idx = np.random.choice(len(self.W), self.N, replace=(len(self.W) < self.N))
            self.W = self.W[idx]

        self.max_fe = max_fe
        self.fe_count = 0

        # Archives
        self.CA_X = None   # (N, n_var)
        self.CA_F = None   # (N, m)
        self.CA_G = None   # (N, n_con)
        self.CA_CV = None  # (N,)

        self.DA_X = None
        self.DA_F = None
        self.DA_G = None
        self.DA_CV = None

        # Ideal / nadir tracking
        self.z_ideal = None
        self.z_nadir = None

    # ─────────────────────────────────────────────────────────────────────────
    # Public API
    # ─────────────────────────────────────────────────────────────────────────

    def run(self):
        """
        Run C-TAEA until max_fe is exhausted.

        Returns
        -------
        CA_X : np.ndarray, shape (N, n_var)  final CA decision variables
        CA_F : np.ndarray, shape (N, m)      final CA objectives
        CA_CV: np.ndarray, shape (N,)        final CA constraint violations
        """
        self._initialize()
        gen = 0

        while self.fe_count < self.max_fe:
            # Generate offspring (2 parents -> 2 children, repeat N times)
            Q_X, Q_F, Q_G, Q_CV = self._generate_offspring()

            # Update ideal/nadir from combined pool
            all_F = np.vstack([self.CA_F, self.DA_F, Q_F])
            self._update_ideal_nadir(all_F)

            # Update CA
            self._update_CA(Q_X, Q_F, Q_G, Q_CV)

            # Update DA (uses updated CA as reference)
            self._update_DA(Q_X, Q_F, Q_G, Q_CV)

            gen += 1
            if self.verbose and gen % 50 == 0:
                n_feasible = int(np.sum(self.CA_CV == 0))
                print(f"  Gen {gen:4d} | FE {self.fe_count:7d} | "
                      f"CA feasible: {n_feasible}/{self.N}")

        return self.CA_X, self.CA_F, self.CA_CV

    # ─────────────────────────────────────────────────────────────────────────
    # Initialization
    # ─────────────────────────────────────────────────────────────────────────

    def _initialize(self):
        """Randomly initialize CA and DA, each of size N."""
        X = np.random.uniform(self.xl, self.xu, size=(2 * self.N, self.n_var))
        F, G = self.problem.evaluate(X)
        CV = self.problem.constraint_violation(G)
        self.fe_count += 2 * self.N

        self._update_ideal_nadir(F)

        # CA: first N, DA: last N (both random initially)
        self.CA_X, self.CA_F, self.CA_G, self.CA_CV = (
            X[:self.N], F[:self.N], G[:self.N], CV[:self.N])
        self.DA_X, self.DA_F, self.DA_G, self.DA_CV = (
            X[self.N:], F[self.N:], G[self.N:], CV[self.N:])

    def _update_ideal_nadir(self, F):
        if self.z_ideal is None:
            self.z_ideal = np.min(F, axis=0)
            self.z_nadir = np.max(F, axis=0)
        else:
            self.z_ideal = np.minimum(self.z_ideal, np.min(F, axis=0))
            self.z_nadir = np.maximum(self.z_nadir, np.max(F, axis=0))

    def _normalize(self, F):
        return normalize_objectives(F, self.z_ideal, self.z_nadir)

    # ─────────────────────────────────────────────────────────────────────────
    # Offspring reproduction (Algorithm 4 + 5)
    # ─────────────────────────────────────────────────────────────────────────

    def _generate_offspring(self):
        """
        Generate N offspring using restricted mating selection.
        Returns Q_X, Q_F, Q_G, Q_CV each of shape (N, ...).
        """
        Q_X = np.zeros((self.N, self.n_var))

        for i in range(0, self.N, 2):
            p1_x = self._restricted_mating_selection()
            p2_x = self._get_second_parent()

            c1_x, c2_x = sbx_crossover(
                p1_x, p2_x, self.xl, self.xu,
                eta_c=self.eta_c, pc=self.pc)
            c1_x = polynomial_mutation(c1_x, self.xl, self.xu, eta_m=self.eta_m)
            c2_x = polynomial_mutation(c2_x, self.xl, self.xu, eta_m=self.eta_m)

            Q_X[i] = c1_x
            if i + 1 < self.N:
                Q_X[i + 1] = c2_x

        Q_F, Q_G = self.problem.evaluate(Q_X)
        Q_CV = self.problem.constraint_violation(Q_G)
        self.fe_count += self.N

        return Q_X, Q_F, Q_G, Q_CV

    def _restricted_mating_selection(self):
        """
        Algorithm 4: Restricted Mating Selection.
        Returns a single parent decision vector.
        """
        rho_c, rho_d = self._compute_rho()

        # Choose first parent
        if rho_c > rho_d:
            p1_x = self._tournament_selection(
                self.CA_X, self.CA_F, self.CA_CV)
        else:
            p1_x = self._tournament_selection(
                self.DA_X, self.DA_F, self.DA_CV)

        return p1_x

    def _compute_rho(self):
        """Compute rho_c and rho_d in combined Hm = CA ∪ DA."""
        all_F = np.vstack([self.CA_F, self.DA_F])
        nd_mask = non_dominated_indices(all_F)
        nd_total = max(np.sum(nd_mask), 1)
        nd_CA_count = np.sum(nd_mask[:self.N])
        nd_DA_count = np.sum(nd_mask[self.N:])
        rho_c = nd_CA_count / nd_total
        rho_d = nd_DA_count / nd_total
        return rho_c, rho_d

    def _get_second_parent(self):
        """Choose second parent based on rho_c."""
        rho_c, _ = self._compute_rho()

        if np.random.rand() < rho_c:
            return self._tournament_selection(
                self.CA_X, self.CA_F, self.CA_CV)
        else:
            return self._tournament_selection(
                self.DA_X, self.DA_F, self.DA_CV)

    def _tournament_selection(self, X, F, CV):
        """
        Algorithm 5: Binary tournament selection (feasibility-driven).
        """
        idx1, idx2 = np.random.choice(len(X), 2, replace=False)
        x1, f1, cv1 = X[idx1], F[idx1], CV[idx1]
        x2, f2, cv2 = X[idx2], F[idx2], CV[idx2]

        feas1 = (cv1 == 0)
        feas2 = (cv2 == 0)

        if feas1 and feas2:
            if dominates(f1, f2):
                return x1
            elif dominates(f2, f1):
                return x2
            else:
                return x1 if np.random.rand() < 0.5 else x2
        elif feas1:
            return x1
        elif feas2:
            return x2
        else:
            # Both infeasible: random
            return x1 if np.random.rand() < 0.5 else x2

    # ─────────────────────────────────────────────────────────────────────────
    # CA Update (Algorithm 2)
    # ─────────────────────────────────────────────────────────────────────────

    def _update_CA(self, Q_X, Q_F, Q_G, Q_CV):
        """
        Algorithm 2: Update Mechanism of the CA.
        """
        N = self.N

        # Combine CA + offspring
        H_X = np.vstack([self.CA_X, Q_X])
        H_F = np.vstack([self.CA_F, Q_F])
        H_G = np.vstack([self.CA_G, Q_G])
        H_CV = np.concatenate([self.CA_CV, Q_CV])

        # Feasible subset Sc
        feas_mask = (H_CV == 0)
        Sc_X = H_X[feas_mask]
        Sc_F = H_F[feas_mask]
        Sc_G = H_G[feas_mask]
        Sc_CV = H_CV[feas_mask]

        n_feas = len(Sc_X)

        if n_feas == N:
            # Lines 5-6
            self.CA_X, self.CA_F, self.CA_G, self.CA_CV = Sc_X, Sc_F, Sc_G, Sc_CV

        elif n_feas > N:
            # Lines 7-21: non-dominated sorting + density trimming
            S_X, S_F, S_G, S_CV = self._select_by_nondom_and_density(
                Sc_X, Sc_F, Sc_G, Sc_CV, N)
            self.CA_X, self.CA_F, self.CA_G, self.CA_CV = S_X, S_F, S_G, S_CV

        else:
            # Lines 23-30: fill with infeasible, sorted by bi-objective (CV, g_tch)
            if n_feas > 0:
                S_X = list(Sc_X)
                S_F = list(Sc_F)
                S_G = list(Sc_G)
                S_CV = list(Sc_CV)
            else:
                S_X, S_F, S_G, S_CV = [], [], [], []

            need = N - len(S_X)

            # Infeasible solutions from Hc
            inf_mask = ~feas_mask
            SI_X = H_X[inf_mask]
            SI_F = H_F[inf_mask]
            SI_G = H_G[inf_mask]
            SI_CV = H_CV[inf_mask]

            if len(SI_X) > 0 and need > 0:
                # Build bi-objective: (CV, g_tch)
                F_norm = self._normalize(SI_F)
                # For g_tch, use best matching weight vector for each solution
                assignments = associate_to_subregions(
                    np.maximum(F_norm, 1e-10), self.W)
                g_tch_vals = np.array([
                    tchebycheff(
                        F_norm[k:k+1],
                        self.W[assignments[k]],
                        z_ideal_norm=None
                    )[0]
                    for k in range(len(SI_F))
                ])

                bi_obj = np.column_stack([SI_CV, g_tch_vals])
                fronts, _ = fast_non_dominated_sort(bi_obj)

                chosen_idx = []
                for front in fronts:
                    if len(chosen_idx) + len(front) <= need:
                        chosen_idx.extend(front)
                    else:
                        # Trim by largest CV first
                        remaining = need - len(chosen_idx)
                        front_cv = SI_CV[front]
                        sorted_by_cv = [front[j] for j in np.argsort(front_cv)]
                        chosen_idx.extend(sorted_by_cv[:remaining])
                    if len(chosen_idx) >= need:
                        break

                chosen_idx = chosen_idx[:need]
                S_X.extend(SI_X[chosen_idx].tolist())
                S_F.extend(SI_F[chosen_idx].tolist())
                S_G.extend(SI_G[chosen_idx].tolist())
                S_CV.extend(SI_CV[chosen_idx].tolist())

            self.CA_X = np.array(S_X)
            self.CA_F = np.array(S_F)
            self.CA_G = np.array(S_G)
            self.CA_CV = np.array(S_CV)

            # Pad if still short (shouldn't happen in practice)
            while len(self.CA_X) < N:
                rand_idx = np.random.randint(len(H_X))
                self.CA_X = np.vstack([self.CA_X, H_X[rand_idx:rand_idx+1]])
                self.CA_F = np.vstack([self.CA_F, H_F[rand_idx:rand_idx+1]])
                self.CA_G = np.vstack([self.CA_G, H_G[rand_idx:rand_idx+1]])
                self.CA_CV = np.append(self.CA_CV, H_CV[rand_idx])

    def _select_by_nondom_and_density(self, X, F, G, CV, N):
        """
        Lines 8-21 of Algorithm 2:
        Non-dominated sorting, then density-based trimming.
        """
        fronts, _ = fast_non_dominated_sort(F)

        S_idx = []
        last_front_i = 0
        for fi, front in enumerate(fronts):
            if len(S_idx) + len(front) <= N:
                S_idx.extend(front)
                last_front_i = fi
            else:
                # Add partial last front
                remaining = N - len(S_idx)
                # Density-based trimming: iteratively remove worst in most crowded region
                partial_idx = list(front)
                full_idx = S_idx + partial_idx

                F_norm_all = self._normalize(F[full_idx])
                assignments = associate_to_subregions(
                    np.maximum(F_norm_all, 1e-10), self.W)

                # Map to local indices
                local_assign = assignments[len(S_idx):]  # for partial front
                all_assign = assignments

                # We trim from partial_idx (NOT from S_idx)
                # We need to trim until total = N
                to_remove = len(full_idx) - N
                current_partial = partial_idx[:]
                current_all_assign = all_assign.copy()

                for _ in range(to_remove):
                    # Find most crowded subregion among partial
                    # Subregion counts over ALL selected (S_idx + current_partial)
                    partial_global = S_idx + current_partial
                    F_norm_curr = self._normalize(F[partial_global])
                    assign_curr = associate_to_subregions(
                        np.maximum(F_norm_curr, 1e-10), self.W)

                    # Count solutions per subregion
                    counts = np.zeros(self.N, dtype=int)
                    for a in assign_curr:
                        if a < self.N:
                            counts[a] += 1

                    # Find most crowded subregion
                    max_count = np.max(counts)
                    crowded_regions = np.where(counts == max_count)[0]
                    # Break ties randomly
                    chosen_region = crowded_regions[np.random.randint(len(crowded_regions))]

                    # Among partial front solutions in that region, find worst
                    # (smallest nearest-neighbor distance, broken by max tchebycheff)
                    partial_in_region = [
                        current_partial[k]
                        for k, a in enumerate(
                            assign_curr[len(S_idx):len(S_idx)+len(current_partial)]
                        )
                        if a == chosen_region
                    ]

                    if len(partial_in_region) == 0:
                        # Fall back: remove a random partial solution in crowded region
                        all_in_region = [
                            partial_global[k]
                            for k, a in enumerate(assign_curr)
                            if a == chosen_region and partial_global[k] not in S_idx
                        ]
                        if all_in_region:
                            partial_in_region = all_in_region
                        else:
                            partial_in_region = [current_partial[0]]

                    xw = self._find_worst_in_region(
                        partial_in_region, F, chosen_region)
                    current_partial.remove(xw)

                S_idx.extend(current_partial)
                break

        if len(S_idx) < N:
            # Should not happen, but safety fallback
            remaining = [i for i in range(len(F)) if i not in S_idx]
            S_idx.extend(remaining[:N - len(S_idx)])

        S_idx = S_idx[:N]
        return X[S_idx], F[S_idx], G[S_idx], CV[S_idx]

    def _find_worst_in_region(self, region_idx, F, w_idx):
        """
        Lines 17-21 of Algorithm 2.
        Among solutions in the subregion, find worst:
          - Compute nearest-neighbor distances
          - Collect those with smallest distance (St)
          - From St, pick max tchebycheff
        """
        if len(region_idx) == 1:
            return region_idx[0]

        F_region = F[region_idx]
        n_r = len(F_region)

        # Nearest neighbor distances within region
        dists = np.full(n_r, np.inf)
        for i in range(n_r):
            for j in range(n_r):
                if i != j:
                    d = np.linalg.norm(F_region[i] - F_region[j])
                    dists[i] = min(dists[i], d)

        min_dist = np.min(dists)
        St_local = np.where(dists == min_dist)[0]  # local indices in St

        if w_idx >= len(self.W):
            w_idx = w_idx % len(self.W)
        w = self.W[w_idx]

        F_norm_region = self._normalize(F_region)
        tch_vals = tchebycheff(F_norm_region[St_local], w)
        worst_local = St_local[np.argmax(tch_vals)]

        return region_idx[worst_local]

    # ─────────────────────────────────────────────────────────────────────────
    # DA Update (Algorithm 3)
    # ─────────────────────────────────────────────────────────────────────────

    def _update_DA(self, Q_X, Q_F, Q_G, Q_CV):
        """
        Algorithm 3: Update Mechanism of the DA.

        DA ignores feasibility; uses up-to-date CA as reference.
        Iteratively fills subregions: at iteration itr, at most itr solutions
        (CA + Hd combined) per subregion.
        """
        N = self.N

        # Hd = DA ∪ Q
        Hd_X = np.vstack([self.DA_X, Q_X])
        Hd_F = np.vstack([self.DA_F, Q_F])
        Hd_G = np.vstack([self.DA_G, Q_G])
        Hd_CV = np.concatenate([self.DA_CV, Q_CV])

        # Associate Hd to subregions
        F_norm_Hd = self._normalize(Hd_F)
        assign_Hd = associate_to_subregions(
            np.maximum(F_norm_Hd, 1e-10), self.W)

        # Associate CA to subregions
        F_norm_CA = self._normalize(self.CA_F)
        assign_CA = associate_to_subregions(
            np.maximum(F_norm_CA, 1e-10), self.W)

        # Build per-subregion lists
        Hd_subregions = [[] for _ in range(N)]  # indices into Hd
        for k, a in enumerate(assign_Hd):
            if a < N:
                Hd_subregions[a].append(k)

        CA_counts = np.zeros(N, dtype=int)
        for a in assign_CA:
            if a < N:
                CA_counts[a] += 1

        # Iterative filling
        S_idx = []   # indices into Hd
        itr = 1

        while len(S_idx) < N:
            added_this_round = False
            for i in range(N):
                if len(S_idx) >= N:
                    break

                # Slots available in subregion i at this iteration
                slots_needed = itr - CA_counts[i]

                if slots_needed <= 0:
                    continue

                for _ in range(slots_needed):
                    # Find non-dominated solutions in Hd_subregions[i]
                    # that haven't been selected yet
                    available = [k for k in Hd_subregions[i] if k not in S_idx]
                    if not available:
                        break

                    Oi = self._non_dominated_subset(available, Hd_F)
                    if not Oi:
                        Oi = available  # fallback: use all available

                    # xb = argmin tchebycheff in Oi
                    F_norm_Oi = self._normalize(Hd_F[Oi])
                    w = self.W[i]
                    tch_vals = tchebycheff(F_norm_Oi, w)
                    best_local = np.argmin(tch_vals)
                    best_idx = Oi[best_local]

                    S_idx.append(best_idx)
                    added_this_round = True

                    if len(S_idx) >= N:
                        break

            itr += 1

            if not added_this_round:
                # Fill remaining from Hd (best tch overall)
                remaining_needed = N - len(S_idx)
                if remaining_needed > 0:
                    not_selected = [k for k in range(len(Hd_X)) if k not in S_idx]
                    if not_selected:
                        F_norm_rem = self._normalize(Hd_F[not_selected])
                        # Use average weight vector
                        w_avg = np.ones(self.m) / self.m
                        tch_vals = tchebycheff(F_norm_rem, w_avg)
                        order = np.argsort(tch_vals)
                        for j in order[:remaining_needed]:
                            S_idx.append(not_selected[j])
                break

        S_idx = S_idx[:N]
        # Ensure uniqueness
        seen = set()
        unique_idx = []
        for k in S_idx:
            if k not in seen:
                seen.add(k)
                unique_idx.append(k)
        # Pad if needed
        if len(unique_idx) < N:
            not_sel = [k for k in range(len(Hd_X)) if k not in seen]
            unique_idx.extend(not_sel[:N - len(unique_idx)])
        unique_idx = unique_idx[:N]

        self.DA_X = Hd_X[unique_idx]
        self.DA_F = Hd_F[unique_idx]
        self.DA_G = Hd_G[unique_idx]
        self.DA_CV = Hd_CV[unique_idx]

    def _non_dominated_subset(self, idx_list, F):
        """Return indices (from idx_list) that are non-dominated in F[idx_list]."""
        if len(idx_list) <= 1:
            return idx_list

        sub_F = F[idx_list]
        nd_mask = non_dominated_indices(sub_F)
        return [idx_list[k] for k in range(len(idx_list)) if nd_mask[k]]

    # ─────────────────────────────────────────────────────────────────────────
    # Result extraction
    # ─────────────────────────────────────────────────────────────────────────

    def get_feasible_CA(self):
        """Return only feasible solutions from CA."""
        feas = self.CA_CV == 0
        return self.CA_X[feas], self.CA_F[feas], self.CA_CV[feas]

    def get_nondominated_CA(self):
        """Return non-dominated feasible solutions from CA."""
        X, F, CV = self.get_feasible_CA()
        if len(F) == 0:
            return X, F, CV
        nd = non_dominated_indices(F)
        return X[nd], F[nd], CV[nd]
