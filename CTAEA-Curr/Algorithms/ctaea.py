

"""
C-TAEA: Two-Archive Evolutionary Algorithm for Constrained Multiobjective Optimization.

Algorithm flow:
1. Initialize CA and DA (each size N)
2. For each generation:
   a. Restricted mating selection -> offspring Q
   b. Update CA using Algorithm 2
   c. Update DA using Algorithm 3
3. Return CA as final result

Implementation notes
--------------------
Hot paths use NumPy vectorization (non-dominated sorting, subregion counts,
Tchebycheff scalars, NN distances in crowded regions). SBX/mutation can use
either a fast batched implementation or row-wise scalar operators that consume
random numbers in the same order as sequential ``sbx_crossover`` /
``polynomial_mutation`` calls (set ``fast_sbx_mutation=False`` for reproducible
runs with a fixed ``seed`` relative to that scalar sequence).
"""


import numpy as np
from .utils import (get_N_and_H, generate_reference_vectors_table_iv,
    normalize_objectives, associate_to_subregions,
    fast_non_dominated_sort, non_dominated_indices,
    sbx_crossover_batch, polynomial_mutation_batch,
    sbx_crossover_match_scalar_rng, polynomial_mutation_match_scalar_rng)


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
        Maximum function evaluations. If None, uses ``1000 * N`` (use
        ``Experiments.runner.get_max_fe`` for paper Table V per instance).
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
    fast_sbx_mutation : bool
        If True (default), use vectorized batched SBX and mutation (faster).
        If False, use row-wise scalar operators; RNG stream matches ``K`` sequential
        ``sbx_crossover`` / ``polynomial_mutation`` calls in pair order.
    use_vectorized_ops : bool or None
        Deprecated alias: if not ``None``, overrides ``fast_sbx_mutation``.
    collect_metrics_stages : sequence of float, optional
        Fractions in (0, 1] of ``max_fe`` at which to append IGD/HV of the current
        non-dominated feasible CA to ``metrics_trace`` (e.g. ``(0.25, 0.5, 1.0)``).
    hv_ref_point : (m,) array, optional
        Reference point for hypervolume at checkpoints (should match experiments).
    metrics_p_star_points : int
        Reference PF sample size for IGD at checkpoints.
    metrics_hv_mc_samples : int
        Monte Carlo samples for HV when ``m > 5`` at checkpoints.
    """

    def __init__(self, problem, N=None, max_fe=None,
                 eta_c=30.0, eta_m=20.0, pc=0.9, seed=None, verbose=False,
                 fast_sbx_mutation=True, use_vectorized_ops=None,
                 collect_metrics_stages=None, hv_ref_point=None,
                 metrics_p_star_points=500, metrics_hv_mc_samples=50000):
        self.problem = problem
        self.m = problem.n_obj
        self.n_var = problem.n_var
        self.xl = problem.xl
        self.xu = problem.xu
        self.eta_c = eta_c
        self.eta_m = eta_m
        self.pc = pc
        self.verbose = verbose
        if use_vectorized_ops is not None:
            fast_sbx_mutation = bool(use_vectorized_ops)
        self.fast_sbx_mutation = fast_sbx_mutation

        if collect_metrics_stages:
            self.collect_metrics_stages = tuple(
                sorted({round(float(x), 12) for x in collect_metrics_stages
                        if 0.0 < float(x) <= 1.0}))
        else:
            self.collect_metrics_stages = ()
        self.hv_ref_point = (np.asarray(hv_ref_point, dtype=float)
                             if hv_ref_point is not None else None)
        self.metrics_p_star_points = int(metrics_p_star_points)
        self.metrics_hv_mc_samples = int(metrics_hv_mc_samples)
        self.metrics_trace = []

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

        if max_fe is None:
            self.max_fe = 1000 * self.N
        else:
            self.max_fe = int(max_fe)

        self._p_star_cached = None
        self._metric_stage_idx = 0

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

        If ``collect_metrics_stages`` was set, see ``metrics_trace`` for IGD/HV
        snapshots at the requested budget fractions.
        """
        self.metrics_trace = []
        self._p_star_cached = None
        self._metric_stage_idx = 0

        self._initialize()
        self._maybe_record_metric_stages()
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
            self._maybe_record_metric_stages()
            if self.verbose and gen % 50 == 0:
                n_feasible = int(np.sum(self.CA_CV == 0))
                print(f"  Gen {gen:4d} | FE {self.fe_count:7d} | "
                      f"CA feasible: {n_feasible}/{self.N}")

        # Log any remaining stages (e.g. 1.0) using the final population
        self._maybe_record_metric_stages(at_end=True)

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

    def _maybe_record_metric_stages(self, at_end=False):
        """Append checkpoint rows when FE fraction crosses ``collect_metrics_stages``."""
        if not self.collect_metrics_stages:
            return
        p = min(1.0, self.fe_count / max(self.max_fe, 1))
        while self._metric_stage_idx < len(self.collect_metrics_stages):
            sf = self.collect_metrics_stages[self._metric_stage_idx]
            if not at_end and p + 1e-12 < sf:
                break
            self._append_metrics_checkpoint(sf)
            self._metric_stage_idx += 1

    def _append_metrics_checkpoint(self, stage_fraction):
        from Metrics.igd import igd
        from Metrics.hv import hypervolume, hypervolume_monte_carlo

        _, F_nd, _ = self.get_nondominated_CA()
        n_nd = len(F_nd)
        if self._p_star_cached is None:
            self._p_star_cached = self.problem.get_pareto_front_reference(
                n_points=self.metrics_p_star_points)
        if n_nd > 0:
            igd_val = float(igd(F_nd, self._p_star_cached))
        else:
            igd_val = float('inf')

        if self.hv_ref_point is not None:
            ref = self.hv_ref_point
        else:
            ref = np.full(self.m, 1.1)

        if n_nd == 0:
            hv_val = 0.0
        else:
            mo = F_nd.shape[1]
            if mo <= 5:
                try:
                    hv_val = float(hypervolume(F_nd, ref))
                except Exception:
                    hv_val = float(hypervolume_monte_carlo(
                        F_nd, ref, n_samples=self.metrics_hv_mc_samples))
            else:
                hv_val = float(hypervolume_monte_carlo(
                    F_nd, ref, n_samples=self.metrics_hv_mc_samples))

        self.metrics_trace.append({
            'fe_fraction': float(stage_fraction),
            'fe': int(self.fe_count),
            'igd': igd_val,
            'hv': hv_val,
            'n_feasible_nd': int(n_nd),
        })

    # ─────────────────────────────────────────────────────────────────────────
    # Offspring reproduction (Algorithm 4 + 5)
    # ─────────────────────────────────────────────────────────────────────────

    def _generate_offspring(self):
        """
        Generate N offspring using restricted mating selection.
        Returns Q_X, Q_F, Q_G, Q_CV each of shape (N, ...).
        """
        rho_c, rho_d = self._compute_rho()
        n_pairs = (self.N + 1) // 2
        P1 = np.empty((n_pairs, self.n_var))
        P2 = np.empty((n_pairs, self.n_var))
        for ip in range(n_pairs):
            P1[ip] = self._restricted_mating_selection(rho_c, rho_d)
            P2[ip] = self._get_second_parent(rho_c)

        if self.fast_sbx_mutation:
            C1, C2 = sbx_crossover_batch(
                P1, P2, self.xl, self.xu, eta_c=self.eta_c, pc=self.pc)
            C1 = polynomial_mutation_batch(
                C1, self.xl, self.xu, eta_m=self.eta_m)
            C2 = polynomial_mutation_batch(
                C2, self.xl, self.xu, eta_m=self.eta_m)
        else:
            C1, C2 = sbx_crossover_match_scalar_rng(
                P1, P2, self.xl, self.xu, eta_c=self.eta_c, pc=self.pc)
            C1 = polynomial_mutation_match_scalar_rng(
                C1, self.xl, self.xu, eta_m=self.eta_m)
            C2 = polynomial_mutation_match_scalar_rng(
                C2, self.xl, self.xu, eta_m=self.eta_m)

        Q_X = np.empty((self.N, self.n_var))
        Q_X[0::2] = C1
        if self.N % 2:
            Q_X[1::2] = C2[: self.N // 2]
        else:
            Q_X[1::2] = C2

        Q_F, Q_G = self.problem.evaluate(Q_X)
        Q_CV = self.problem.constraint_violation(Q_G)
        self.fe_count += self.N

        return Q_X, Q_F, Q_G, Q_CV

    def _restricted_mating_selection(self, rho_c, rho_d):
        """
        Algorithm 4: Restricted Mating Selection.
        Returns a single parent decision vector.
        """
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

    def _get_second_parent(self, rho_c):
        """Choose second parent based on rho_c."""
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
            d12 = np.all(f1 <= f2) and np.any(f1 < f2)
            d21 = np.all(f2 <= f1) and np.any(f2 < f1)
            if d12:
                return x1
            elif d21:
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
                w_sel = np.maximum(self.W[assignments], 1e-6)
                g_tch_vals = np.max(F_norm / w_sel, axis=1)

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
        for front in fronts:
            if len(S_idx) + len(front) <= N:
                S_idx.extend(front)
            else:
                # Add partial last front
                remaining = N - len(S_idx)
                # Density-based trimming: iteratively remove worst in most crowded region
                partial_idx = list(front)
                full_idx = S_idx + partial_idx

                F_norm_all = self._normalize(F[full_idx])
                assignments = associate_to_subregions(
                    np.maximum(F_norm_all, 1e-10), self.W)

                # We trim from partial_idx (NOT from S_idx)
                # We need to trim until total = N
                to_remove = len(full_idx) - N
                current_partial = partial_idx[:]

                for _ in range(to_remove):
                    # Find most crowded subregion among partial
                    # Subregion counts over ALL selected (S_idx + current_partial)
                    partial_global = S_idx + current_partial
                    F_norm_curr = self._normalize(F[partial_global])
                    assign_curr = associate_to_subregions(
                        np.maximum(F_norm_curr, 1e-10), self.W)

                    valid_a = assign_curr < self.N
                    counts = np.bincount(
                        assign_curr[valid_a], minlength=self.N)

                    # Find most crowded subregion
                    max_count = np.max(counts)
                    crowded_regions = np.where(counts == max_count)[0]
                    # Break ties randomly
                    chosen_region = crowded_regions[np.random.randint(len(crowded_regions))]

                    # Among partial front solutions in that region, find worst
                    # (smallest nearest-neighbor distance, broken by max tchebycheff)
                    off = len(S_idx)
                    plen = len(current_partial)
                    p_assign = assign_curr[off : off + plen]
                    pr_arr = np.asarray(current_partial, dtype=int)
                    partial_in_region = pr_arr[p_assign == chosen_region].tolist()

                    if len(partial_in_region) == 0:
                        # Fall back: remove a random partial solution in crowded region
                        pg = np.asarray(partial_global, dtype=int)
                        mask = (assign_curr == chosen_region) & ~np.isin(
                            pg, np.asarray(S_idx, dtype=int))
                        all_in_region = pg[mask].tolist()
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

        idx = np.asarray(region_idx, dtype=int)
        F_region = F[idx]
        D = np.linalg.norm(
            F_region[:, None, :] - F_region[None, :, :], axis=2)
        np.fill_diagonal(D, np.inf)
        dists = D.min(axis=1)

        min_dist = np.min(dists)
        St_local = np.where(dists == min_dist)[0]

        if w_idx >= len(self.W):
            w_idx = w_idx % len(self.W)
        w = self.W[w_idx]

        F_norm_region = self._normalize(F_region)
        Fn_st = F_norm_region[St_local]
        w_safe = np.where(w < 1e-6, 1e-6, w)
        tch_vals = np.max(np.abs(Fn_st) / w_safe, axis=1)
        worst_local = St_local[int(np.argmax(tch_vals))]

        return int(idx[worst_local])

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

        ca_ok = assign_CA < N
        CA_counts = np.bincount(assign_CA[ca_ok], minlength=N)

        # Iterative filling
        S_idx = []   # indices into Hd
        S_set = set()
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
                    avail_list = Hd_subregions[i]
                    available = [k for k in avail_list if k not in S_set]
                    if not available:
                        break

                    Oi = self._non_dominated_subset(available, Hd_F)
                    if not Oi:
                        Oi = available  # fallback: use all available

                    # xb = argmin tchebycheff in Oi
                    F_norm_Oi = self._normalize(Hd_F[Oi])
                    w = self.W[i]
                    w_safe = np.where(w < 1e-6, 1e-6, w)
                    tch_vals = np.max(np.abs(F_norm_Oi) / w_safe, axis=1)
                    best_local = int(np.argmin(tch_vals))
                    best_idx = Oi[best_local]

                    S_idx.append(best_idx)
                    S_set.add(best_idx)
                    added_this_round = True

                    if len(S_idx) >= N:
                        break

            itr += 1

            if not added_this_round:
                # Fill remaining from Hd (best tch overall)
                remaining_needed = N - len(S_idx)
                if remaining_needed > 0:
                    not_selected = sorted(
                        set(range(len(Hd_X))) - S_set)
                    if not_selected:
                        F_norm_rem = self._normalize(Hd_F[not_selected])
                        # Use average weight vector
                        w_avg = np.ones(self.m) / self.m
                        w_safe = np.where(w_avg < 1e-6, 1e-6, w_avg)
                        tch_vals = np.max(np.abs(F_norm_rem) / w_safe, axis=1)
                        order = np.argsort(tch_vals)
                        for j in order[:remaining_needed]:
                            kk = not_selected[j]
                            S_idx.append(kk)
                            S_set.add(kk)
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
            not_sel = sorted(set(range(len(Hd_X))) - seen)
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

        idx_arr = np.asarray(idx_list, dtype=int)
        nd_mask = non_dominated_indices(F[idx_arr])
        return idx_arr[nd_mask].tolist()

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
