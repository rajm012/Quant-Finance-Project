"""
Parallel experiment runner for C-TAEA replication.

Strategy for the AMD EPYC 9124 + 8x RTX A6000 server:
  - The algorithms are pure NumPy / CPU-bound, so we use multiprocessing.
  - We build a flat task list of (problem, m, algo, run_idx, seed) tuples,
    then drain them with a ProcessPoolExecutor using ~56 workers.
  - Results are flushed to disk after every task so a crash loses at most
    one job, and a subsequent run automatically skips completed tasks.
  - NUMA / CPU affinity: each worker is pinned to one logical core via
    os.sched_setaffinity when the OS supports it (Linux).

Usage
-----
  # Full replication (51 runs × 11 problems × 5 m-values × 6 algorithms):
  python parallel_runner.py

  # Tune parallelism (default 56 workers):
  python parallel_runner.py --workers 48

  # Quick smoke test (3 runs, m=3 only):
  python parallel_runner.py --quick

  # Resume a previously interrupted run:
  python parallel_runner.py --resume

  # Specific subset:
  python parallel_runner.py --problems C1DTLZ3 C2DTLZ2 --m 3 5 --algos C-TAEA C-NSGA-III
"""

import sys
import os
import argparse
import json
import time
import hashlib
import signal
import logging
import traceback
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

# ── make sure project root is importable inside worker processes ──────────────
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

# ── logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("parallel_runner")


# ─────────────────────────────────────────────────────────────────────────────
# Experiment configuration  (mirrors runner.py)
# ─────────────────────────────────────────────────────────────────────────────

FE_TABLE = {
    "C1DTLZ1":  {3: 500,  5: 600,  8: 800,  10: 1000, 15: 1500},
    "C1DTLZ3":  {3: 1000, 5: 1500, 8: 2500, 10: 3500, 15: 5000},
    "C2DTLZ2":  {3: 250,  5: 350,  8: 500,  10: 750,  15: 1000},
    "C3DTLZ1":  {3: 750,  5: 1250, 8: 2000, 10: 3000, 15: 4000},
    "C3DTLZ4":  {3: 750,  5: 1250, 8: 2000, 10: 3000, 15: 4000},
    "DC1DTLZ1": {3: 500,  5: 600,  8: 800,  10: 1000, 15: 1500},
    "DC1DTLZ3": {3: 1000, 5: 1500, 8: 2500, 10: 3500, 15: 5000},
    "DC2DTLZ1": {3: 500,  5: 600,  8: 800,  10: 1000, 15: 1500},
    "DC2DTLZ3": {3: 1000, 5: 1500, 8: 2500, 10: 3500, 15: 5000},
    "DC3DTLZ1": {3: 750,  5: 1250, 8: 2000, 10: 3000, 15: 4000},
    "DC3DTLZ3": {3: 750,  5: 1250, 8: 2000, 10: 3000, 15: 4000},
}

ZR_TABLE = {
    "C3DTLZ4": 2.1,
    "default":  1.1,
}

ALL_PROBLEMS = list(FE_TABLE.keys())
ALL_M        = [3, 5, 8, 10, 15]
ALL_ALGOS    = ["C-TAEA", "C-NSGA-III", "C-MOEA/D", "C-MOEA/DD", "I-DBEA", "CMOEA"]


# ─────────────────────────────────────────────────────────────────────────────
# Worker function  (must be top-level for pickle/multiprocessing)
# ─────────────────────────────────────────────────────────────────────────────

def _worker(task: dict) -> dict:
    """
    Execute one (problem, m, algo, run_idx) trial.

    Parameters
    ----------
    task : dict with keys
        problem_name, m, algo_name, run_idx, seed

    Returns
    -------
    dict: task fields + igd, hv, n_feasible, time_sec, error
    """
    # ── optional: pin this process to a specific CPU core ────────────────────
    worker_id = task.get("worker_slot", None)
    if worker_id is not None and hasattr(os, "sched_setaffinity"):
        try:
            os.sched_setaffinity(0, {worker_id % os.cpu_count()})
        except OSError:
            pass  # may lack CAP_SYS_NICE — non-fatal

    # ── imports inside worker so forked processes don't share state ──────────
    from Problems import (
        C1DTLZ1, C1DTLZ3, C2DTLZ2, C3DTLZ1, C3DTLZ4,
        DC1DTLZ1, DC1DTLZ3, DC2DTLZ1, DC2DTLZ3, DC3DTLZ1, DC3DTLZ3,
    )
    from Algorithms import CTAEA, CMOEAD, CNSGAIII, CMOEAD_DD, IDBEA, CMOEA
    from Algorithms.utils import get_N_and_H, non_dominated_indices
    from Metrics.igd import igd
    from Metrics.hv import hypervolume, hypervolume_monte_carlo

    PROBLEM_MAP = {
        "C1DTLZ1": C1DTLZ1, "C1DTLZ3": C1DTLZ3, "C2DTLZ2": C2DTLZ2,
        "C3DTLZ1": C3DTLZ1, "C3DTLZ4": C3DTLZ4,
        "DC1DTLZ1": DC1DTLZ1, "DC1DTLZ3": DC1DTLZ3,
        "DC2DTLZ1": DC2DTLZ1, "DC2DTLZ3": DC2DTLZ3,
        "DC3DTLZ1": DC3DTLZ1, "DC3DTLZ3": DC3DTLZ3,
    }
    ALGO_MAP = {
        "C-TAEA":    CTAEA,
        "C-NSGA-III": CNSGAIII,
        "C-MOEA/D":  CMOEAD,
        "C-MOEA/DD": CMOEAD_DD,
        "I-DBEA":    IDBEA,
        "CMOEA":     CMOEA,
    }

    prob_name = task["problem_name"]
    m         = task["m"]
    algo_name = task["algo_name"]
    seed      = task["seed"]

    result = dict(task)  # echo task fields back
    result["igd"] = float("inf")
    result["hv"]  = 0.0
    result["n_feasible"] = 0
    result["time_sec"]   = 0.0
    result["error"]      = None

    try:
        problem = PROBLEM_MAP[prob_name](n_obj=m)
        N, _    = get_N_and_H(m)
        fe_mult = FE_TABLE.get(prob_name, {}).get(m, 500)
        max_fe  = fe_mult * N

        algo = ALGO_MAP[algo_name](
            problem=problem,
            N=N,
            max_fe=max_fe,
            seed=seed,
            verbose=False,
        )

        t0 = time.perf_counter()
        algo.run()
        elapsed = time.perf_counter() - t0

        # ── extract non-dominated feasible front ─────────────────────────────
        if hasattr(algo, "get_nondominated"):
            _, F_nd, _ = algo.get_nondominated()
        elif hasattr(algo, "get_nondominated_CA"):
            _, F_nd, _ = algo.get_nondominated_CA()
        else:
            feas = algo.pop_CV == 0
            F_f  = algo.pop_F[feas]
            if len(F_f) > 0:
                F_nd = F_f[non_dominated_indices(F_f)]
            else:
                F_nd = F_f

        # ── metrics ──────────────────────────────────────────────────────────
        P_star  = problem.get_pareto_front_reference(n_points=500)
        igd_val = igd(F_nd, P_star) if len(F_nd) > 0 else float("inf")

        zr_val  = ZR_TABLE.get(prob_name, ZR_TABLE["default"])
        ref_pt  = np.full(m, zr_val)
        if len(F_nd) > 0:
            try:
                hv_val = hypervolume(F_nd, ref_pt) if m <= 5 else \
                         hypervolume_monte_carlo(F_nd, ref_pt, n_samples=50_000)
            except Exception:
                hv_val = hypervolume_monte_carlo(F_nd, ref_pt, n_samples=50_000)
        else:
            hv_val = 0.0

        result["igd"]        = float(igd_val)
        result["hv"]         = float(hv_val)
        result["n_feasible"] = int(len(F_nd))
        result["time_sec"]   = float(elapsed)

    except Exception:
        result["error"] = traceback.format_exc()

    return result


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _task_key(t: dict) -> str:
    return f"{t['problem_name']}|m{t['m']}|{t['algo_name']}|run{t['run_idx']}"


def _result_path(out_dir: Path, t: dict) -> Path:
    safe_algo = t["algo_name"].replace("/", "_")
    return out_dir / "raw" / \
        f"{t['problem_name']}_m{t['m']}_{safe_algo}_run{t['run_idx']:02d}.json"


def _save_result(out_dir: Path, result: dict) -> None:
    p = _result_path(out_dir, result)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w") as fh:
        json.dump(_to_serializable(result), fh)


def _load_completed(out_dir: Path) -> set:
    """Return the set of task_keys that already have saved results."""
    completed = set()
    raw_dir = out_dir / "raw"
    if not raw_dir.exists():
        return completed
    for p in raw_dir.glob("*.json"):
        try:
            with open(p) as fh:
                d = json.load(fh)
            if d.get("error") is None:          # only count clean results
                completed.add(_task_key(d))
        except Exception:
            pass
    return completed


def _to_serializable(obj):
    if isinstance(obj, dict):
        return {str(k): _to_serializable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_to_serializable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    return obj


# ─────────────────────────────────────────────────────────────────────────────
# Build task list
# ─────────────────────────────────────────────────────────────────────────────

def build_tasks(problem_names, m_values, algo_names, n_runs):
    """Return sorted list of task dicts."""
    tasks = []
    for prob in problem_names:
        # reproducible seed base per (problem, m) — same as original runner.py
        for m in m_values:
            seed_key = f"{prob}|{m}"
            base = int(hashlib.sha256(seed_key.encode()).hexdigest()[:8], 16)
            for algo in algo_names:
                for run_idx in range(n_runs):
                    seed = (base + run_idx * 100) % (2**31)
                    tasks.append({
                        "problem_name": prob,
                        "m":            m,
                        "algo_name":    algo,
                        "run_idx":      run_idx,
                        "seed":         seed,
                    })
    return tasks


# ─────────────────────────────────────────────────────────────────────────────
# Aggregate raw results → summary JSON (mirrors runner.py format)
# ─────────────────────────────────────────────────────────────────────────────

def aggregate_results(out_dir: Path, problem_names, m_values, algo_names, n_runs):
    log.info("Aggregating results …")
    combined = {}

    for prob in problem_names:
        combined[prob] = {}
        for m in m_values:
            combined[prob][m] = {}
            for algo in algo_names:
                run_list = []
                for run_idx in range(n_runs):
                    p = _result_path(out_dir, {
                        "problem_name": prob, "m": m,
                        "algo_name": algo, "run_idx": run_idx,
                    })
                    if p.exists():
                        with open(p) as fh:
                            run_list.append(json.load(fh))
                    else:
                        run_list.append({"igd": float("inf"), "hv": 0.0,
                                         "n_feasible": 0, "error": "missing"})

                igd_vals = [r["igd"] for r in run_list]
                hv_vals  = [r["hv"]  for r in run_list]

                summary = {
                    "igd_median": float(np.median(igd_vals)),
                    "igd_iqr":    float(np.subtract(
                        *np.percentile(igd_vals, [75, 25]))),
                    "hv_median":  float(np.median(hv_vals)),
                    "hv_iqr":     float(np.subtract(
                        *np.percentile(hv_vals, [75, 25]))),
                    "n_runs_feasible": sum(r["n_feasible"] > 0 for r in run_list),
                    "n_runs_total":    len(run_list),
                    "runs": run_list,
                }
                combined[prob][m][algo] = summary

    agg_path = out_dir / "all_results.json"
    with open(agg_path, "w") as fh:
        json.dump(_to_serializable(combined), fh, indent=2)
    log.info("Aggregated results saved → %s", agg_path)
    return combined


# ─────────────────────────────────────────────────────────────────────────────
# Print a concise summary table
# ─────────────────────────────────────────────────────────────────────────────

def print_summary(combined):
    print("\n" + "=" * 80)
    print("RESULTS SUMMARY  (IGD median  |  HV median)")
    print("=" * 80)
    for prob, m_dict in combined.items():
        for m, algo_dict in m_dict.items():
            print(f"\n  {prob}, m={m}")
            for algo, s in algo_dict.items():
                feas = s["n_runs_feasible"]
                tot  = s["n_runs_total"]
                print(f"    {algo:<14s}  IGD={s['igd_median']:.4e}"
                      f"  HV={s['hv_median']:.4e}"
                      f"  feas_runs={feas}/{tot}")
    print("=" * 80 + "\n")


# ─────────────────────────────────────────────────────────────────────────────
# Main parallel experiment loop
# ─────────────────────────────────────────────────────────────────────────────

def run_parallel_experiment(
    problem_names=None,
    m_values=None,
    algo_names=None,
    n_runs=51,
    output_dir="Results_parallel",
    n_workers=56,
    resume=True,
):
    """
    Run the full (or partial) C-TAEA replication experiment in parallel.

    Parameters
    ----------
    problem_names : list[str] | None   subset of ALL_PROBLEMS; None = all
    m_values      : list[int] | None   subset of ALL_M; None = all
    algo_names    : list[str] | None   subset of ALL_ALGOS; None = all
    n_runs        : int                independent runs per configuration
    output_dir    : str                where results are stored
    n_workers     : int                parallel CPU processes
    resume        : bool               skip already-completed tasks
    """
    problem_names = problem_names or ALL_PROBLEMS
    m_values      = m_values      or ALL_M
    algo_names    = algo_names    or ALL_ALGOS

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_tasks = build_tasks(problem_names, m_values, algo_names, n_runs)

    # ── resume: drop already-finished tasks ──────────────────────────────────
    completed = _load_completed(out_dir) if resume else set()
    pending   = [t for t in all_tasks if _task_key(t) not in completed]

    total   = len(all_tasks)
    skipped = total - len(pending)
    log.info("Tasks: %d total | %d completed (skipped) | %d pending",
             total, skipped, len(pending))
    log.info("Workers: %d  |  Output: %s", n_workers, out_dir)

    if not pending:
        log.info("Nothing to do — all tasks already complete.")
        return aggregate_results(out_dir, problem_names, m_values, algo_names, n_runs)

    # ── assign worker_slot for optional CPU pinning ──────────────────────────
    for i, t in enumerate(pending):
        t["worker_slot"] = i % n_workers

    # ── graceful Ctrl-C: let running futures finish ───────────────────────────
    interrupted = False

    def _sigint_handler(sig, frame):
        nonlocal interrupted
        if not interrupted:
            log.warning("Caught SIGINT — finishing in-flight tasks, then stopping …")
            interrupted = True

    signal.signal(signal.SIGINT, _sigint_handler)

    # ── counters ─────────────────────────────────────────────────────────────
    done    = skipped
    errors  = 0
    t_start = time.perf_counter()

    log.info("Submitting %d tasks to %d workers …", len(pending), n_workers)

    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        future_to_task = {executor.submit(_worker, t): t for t in pending}

        for future in as_completed(future_to_task):
            if interrupted:
                # Cancel remaining futures (best-effort)
                for f in future_to_task:
                    f.cancel()
                break

            orig_task = future_to_task[future]
            try:
                result = future.result()
            except Exception as exc:
                errors += 1
                log.error("Task %s raised: %s", _task_key(orig_task), exc)
                result = dict(orig_task)
                result.update(igd=float("inf"), hv=0.0, n_feasible=0,
                              time_sec=0.0, error=str(exc))

            # ── persist immediately ───────────────────────────────────────────
            _save_result(out_dir, result)
            done += 1

            if result.get("error"):
                errors += 1
                log.error("[%4d/%d] FAIL  %s | %s",
                           done, total, _task_key(result),
                           str(result["error"])[:120])
            else:
                elapsed_total = time.perf_counter() - t_start
                rate   = (done - skipped) / max(elapsed_total, 1e-3)
                eta    = (len(pending) - (done - skipped)) / max(rate, 1e-6)
                log.info(
                    "[%4d/%d]  %s | IGD=%.4e HV=%.4e feas=%d  "
                    "t=%.1fs  rate=%.2f/s  ETA=%.0fm",
                    done, total,
                    _task_key(result),
                    result["igd"], result["hv"], result["n_feasible"],
                    result["time_sec"],
                    rate,
                    eta / 60,
                )

    if interrupted:
        log.warning("Run interrupted. %d/%d tasks completed. "
                    "Re-run with --resume to continue.", done, total)
    else:
        wall = time.perf_counter() - t_start
        log.info("All pending tasks done in %.1f min  |  errors=%d",
                 wall / 60, errors)

    return aggregate_results(out_dir, problem_names, m_values, algo_names, n_runs)


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def _parse_args():
    p = argparse.ArgumentParser(
        description="Parallel C-TAEA replication runner",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--workers", type=int, default=56,
                   help="Number of parallel worker processes. "
                        "Recommended 48-60 on the EPYC 9124 (64 threads).")
    p.add_argument("--runs", type=int, default=51,
                   help="Independent runs per configuration (paper uses 51).")
    p.add_argument("--output", type=str, default="Results_parallel",
                   help="Directory to store results.")
    p.add_argument("--problems", nargs="+", default=None,
                   choices=ALL_PROBLEMS, metavar="PROB",
                   help="Subset of problems. Default: all 11.")
    p.add_argument("--m", nargs="+", type=int, default=None,
                   choices=ALL_M, metavar="M",
                   help="Objective counts. Default: 3 5 8 10 15.")
    p.add_argument("--algos", nargs="+", default=None,
                   choices=ALL_ALGOS, metavar="ALGO",
                   help="Algorithms. Default: all 6.")
    p.add_argument("--quick", action="store_true",
                   help="Smoke test: 3 runs, m=3 only, all problems & algos.")
    p.add_argument("--no-resume", dest="resume", action="store_false",
                   help="Re-run from scratch (ignore already saved results).")
    p.set_defaults(resume=True)
    return p.parse_args()


def main():
    args = _parse_args()

    m_values = args.m
    n_runs   = args.runs

    if args.quick:
        m_values = [3]
        n_runs   = 3
        log.info("Quick mode: m=3, 3 runs")

    combined = run_parallel_experiment(
        problem_names=args.problems,
        m_values=m_values,
        algo_names=args.algos,
        n_runs=n_runs,
        output_dir=args.output,
        n_workers=args.workers,
        resume=args.resume,
    )

    print_summary(combined)


if __name__ == "__main__":
    main()
