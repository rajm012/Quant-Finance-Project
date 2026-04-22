"""
Analysis Module
===============
Generates comparison tables matching Tables I and III from the paper.
Performs Wilcoxon rank-sum test at 5% significance level.
Outputs LaTeX and CSV tables.
"""

import numpy as np
import os, sys, json
from scipy import stats
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Problems import C_DTLZ_PROBLEMS, DC_DTLZ_PROBLEMS


OBJECTIVES    = [3, 5, 8, 10, 15]
ALL_ALGOS     = ['C-TAEA', 'C-NSGA-III', 'C-MOEA/D', 'C-MOEA/DD', 'I-DBEA', 'CMOEA']
SIGNIFICANCE  = 0.05


# ---------------------------------------------------------------------------
# Load results
# ---------------------------------------------------------------------------

def load_results(results_dir: str = 'Results') -> dict:
    """Load all result files from the results directory."""
    results = {}
    if not os.path.exists(results_dir):
        print(f"Results directory '{results_dir}' not found.")
        return results

    for fname in os.listdir(results_dir):
        if fname.endswith('.json'):
            fpath = os.path.join(results_dir, fname)
            try:
                with open(fpath) as f:
                    data = json.load(f)
                key = f"{data['algorithm']}_{data['problem']}_m{data['m']}"
                results[key] = data
            except Exception as e:
                print(f"Warning: could not load {fname}: {e}")
    return results


def get_result(results: dict, algo: str, problem: str, m: int) -> dict:
    key = f"{algo}_{problem}_m{m}"
    return results.get(key, None)


# ---------------------------------------------------------------------------
# Wilcoxon rank-sum test
# ---------------------------------------------------------------------------

def wilcoxon_test(vals_a: list, vals_b: list) -> str:
    """
    Wilcoxon rank-sum test: is algorithm a significantly better than b?
    Returns:
      '†' if a is significantly better than b
      '‡' if b is significantly better than a
      ' ' if no significant difference
    """
    a = np.array([v for v in vals_a if v is not None and not np.isnan(v)])
    b = np.array([v for v in vals_b if v is not None and not np.isnan(v)])

    if len(a) < 3 or len(b) < 3:
        return ' '

    try:
        stat, p = stats.ranksums(a, b)
        if p < SIGNIFICANCE:
            if np.median(a) < np.median(b):  # lower IGD is better
                return '†'   # a better
            else:
                return '‡'   # b better
    except Exception:
        pass
    return ' '


def wilcoxon_test_hv(vals_a: list, vals_b: list) -> str:
    """Same as above but for HV (higher is better)."""
    a = np.array([v for v in vals_a if v is not None and not np.isnan(v)])
    b = np.array([v for v in vals_b if v is not None and not np.isnan(v)])

    if len(a) < 3 or len(b) < 3:
        return ' '

    try:
        stat, p = stats.ranksums(a, b)
        if p < SIGNIFICANCE:
            if np.median(a) > np.median(b):  # higher HV is better
                return '†'   # a better (a has higher HV than b)
            else:
                return '‡'   # b better
    except Exception:
        pass
    return ' '


# ---------------------------------------------------------------------------
# Table generation
# ---------------------------------------------------------------------------

def format_val(val, iqr, is_best: bool = False) -> str:
    """Format median(IQR) for table cell."""
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return "N/A"
    if val == np.inf:
        return "∞"
    if abs(val) < 1e-4 and val != 0:
        s = f"{val:.3E}({iqr:.2E})"
    elif abs(val) > 1e4:
        s = f"{val:.4E}({iqr:.2E})"
    else:
        s = f"{val:.4f}({iqr:.4f})" if iqr is not None else f"{val:.4f}"
    return f"**{s}**" if is_best else s


def generate_igd_table(results: dict,
                        problem_names: list,
                        objectives: list = OBJECTIVES,
                        output_path: str = None) -> str:
    """
    Generate IGD comparison table (like Table I and III in the paper).
    """
    lines = []
    header = f"{'Problem':12s} {'m':>3s} | " + \
             " | ".join(f"{a:>22s}" for a in ALL_ALGOS)
    lines.append(header)
    lines.append("-" * len(header))

    for prob in problem_names:
        for m in objectives:
            row_vals = {}
            for algo in ALL_ALGOS:
                res = get_result(results, algo, prob, m)
                if res:
                    row_vals[algo] = {
                        'median': res.get('igd_median'),
                        'iqr':    res.get('igd_iqr'),
                        'all':    res.get('igd_all', []),
                    }

            if not row_vals:
                continue

            # Find best (lowest median IGD)
            valid_medians = {a: row_vals[a]['median']
                             for a in row_vals
                             if row_vals[a]['median'] is not None
                             and not np.isnan(row_vals[a]['median'])}
            if not valid_medians:
                continue
            best_algo = min(valid_medians, key=valid_medians.get)

            cells = []
            ctaea_vals = row_vals.get('C-TAEA', {}).get('all', [])

            for algo in ALL_ALGOS:
                if algo not in row_vals:
                    cells.append(f"{'N/A':>22s}")
                    continue

                v = row_vals[algo]
                med = v['median']
                iqr = v['iqr']
                is_best = (algo == best_algo)

                cell = format_val(med, iqr, is_best)

                # Add Wilcoxon marker for non-C-TAEA algorithms
                if algo != 'C-TAEA' and ctaea_vals and v['all']:
                    marker = wilcoxon_test(ctaea_vals, v['all'])
                    cell += marker

                cells.append(f"{cell:>22s}")

            row = f"{prob:12s} {m:>3d} | " + " | ".join(cells)
            lines.append(row)

        lines.append("")

    table_str = "\n".join(lines)

    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w') as f:
            f.write(table_str)
        print(f"Saved IGD table to {output_path}")

    return table_str


def generate_hv_table(results: dict,
                       problem_names: list,
                       objectives: list = OBJECTIVES,
                       output_path: str = None) -> str:
    """Generate HV comparison table."""
    lines = []
    header = f"{'Problem':12s} {'m':>3s} | " + \
             " | ".join(f"{a:>22s}" for a in ALL_ALGOS)
    lines.append("HV COMPARISON TABLE")
    lines.append(header)
    lines.append("-" * len(header))

    for prob in problem_names:
        for m in objectives:
            row_vals = {}
            for algo in ALL_ALGOS:
                res = get_result(results, algo, prob, m)
                if res:
                    row_vals[algo] = {
                        'median': res.get('hv_median'),
                        'iqr':    res.get('hv_iqr'),
                        'all':    res.get('hv_all', []),
                    }

            if not row_vals:
                continue

            valid_medians = {a: row_vals[a]['median']
                             for a in row_vals
                             if row_vals[a]['median'] is not None
                             and not np.isnan(row_vals[a]['median'])}
            if not valid_medians:
                continue
            best_algo = max(valid_medians, key=valid_medians.get)  # HV: higher=better

            cells = []
            ctaea_vals = row_vals.get('C-TAEA', {}).get('all', [])

            for algo in ALL_ALGOS:
                if algo not in row_vals:
                    cells.append(f"{'N/A':>22s}")
                    continue

                v = row_vals[algo]
                med = v['median']
                iqr = v['iqr']
                is_best = (algo == best_algo)

                cell = format_val(med, iqr, is_best)
                if algo != 'C-TAEA' and ctaea_vals and v['all']:
                    marker = wilcoxon_test_hv(ctaea_vals, v['all'])
                    cell += marker

                cells.append(f"{cell:>22s}")

            row = f"{prob:12s} {m:>3d} | " + " | ".join(cells)
            lines.append(row)

        lines.append("")

    table_str = "\n".join(lines)
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w') as f:
            f.write(table_str)
    return table_str


def generate_feasible_runs_table(results: dict,
                                  problem_names: list,
                                  objectives: list = OBJECTIVES,
                                  output_path: str = None) -> str:
    """
    Table II: number of runs where feasible solutions were found.
    """
    lines = ["FEASIBLE RUNS TABLE (like Table II in paper)"]
    header = f"{'Problem':12s} {'m':>3s} | " + \
             " | ".join(f"{a:>10s}" for a in ALL_ALGOS)
    lines.append(header)
    lines.append("-" * len(header))

    for prob in problem_names:
        for m in objectives:
            cells = []
            for algo in ALL_ALGOS:
                res = get_result(results, algo, prob, m)
                n_runs = res.get('n_runs', 51) if res else 51
                if res:
                    fr = res.get('feasible_runs', 0)
                    cells.append(f"{fr:>4d}/{n_runs:<5d}")
                else:
                    cells.append(f"{'N/A':>10s}")

            row = f"{prob:12s} {m:>3d} | " + " | ".join(cells)
            lines.append(row)
        lines.append("")

    table_str = "\n".join(lines)
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w') as f:
            f.write(table_str)
    return table_str


def generate_summary_stats(results: dict,
                            output_path: str = 'Analysis/summary.json'):
    """
    Compute win/tie/loss counts for C-TAEA vs each peer algorithm.
    """
    from Problems import C_DTLZ_PROBLEMS, DC_DTLZ_PROBLEMS
    all_problems = C_DTLZ_PROBLEMS + DC_DTLZ_PROBLEMS
    peers = [a for a in ALL_ALGOS if a != 'C-TAEA']

    summary = {}
    for peer in peers:
        wins = ties = losses = 0
        for prob in all_problems:
            for m in OBJECTIVES:
                ctaea_res = get_result(results, 'C-TAEA', prob, m)
                peer_res  = get_result(results, peer, prob, m)
                if ctaea_res is None or peer_res is None:
                    continue

                ctaea_vals = ctaea_res.get('igd_all', [])
                peer_vals  = peer_res.get('igd_all', [])

                marker = wilcoxon_test(ctaea_vals, peer_vals)
                if marker == '†':
                    wins += 1
                elif marker == '‡':
                    losses += 1
                else:
                    ties += 1

        summary[peer] = {'wins': wins, 'ties': ties, 'losses': losses}
        print(f"C-TAEA vs {peer}: W={wins} T={ties} L={losses}")

    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(summary, f, indent=2)

    return summary


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--results_dir', default='Results')
    parser.add_argument('--output_dir',  default='Analysis')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    results = load_results(args.results_dir)
    print(f"Loaded {len(results)} result entries.\n")

    if not results:
        print("No results found. Run experiments first with:")
        print("  python Experiments/run_experiments.py --quick")
        sys.exit(0)

    # C-DTLZ Table (Table I)
    print("\n=== IGD TABLE: C-DTLZ ===")
    t = generate_igd_table(
        results, C_DTLZ_PROBLEMS,
        output_path=os.path.join(args.output_dir, 'table_I_igd_cdtlz.txt')
    )
    print(t[:2000])  # print first 2000 chars

    # DC-DTLZ Table (Table III)
    print("\n=== IGD TABLE: DC-DTLZ ===")
    t = generate_igd_table(
        results, DC_DTLZ_PROBLEMS,
        output_path=os.path.join(args.output_dir, 'table_III_igd_dcdtlz.txt')
    )
    print(t[:2000])

    # HV Tables (Tables VI, VII)
    generate_hv_table(
        results, C_DTLZ_PROBLEMS,
        output_path=os.path.join(args.output_dir, 'table_VI_hv_cdtlz.txt')
    )
    generate_hv_table(
        results, DC_DTLZ_PROBLEMS,
        output_path=os.path.join(args.output_dir, 'table_VII_hv_dcdtlz.txt')
    )

    # Feasible runs Table II
    print("\n=== FEASIBLE RUNS TABLE ===")
    t = generate_feasible_runs_table(
        results, DC_DTLZ_PROBLEMS,
        output_path=os.path.join(args.output_dir, 'table_II_feasible_runs.txt')
    )
    print(t[:1000])

    # Summary
    print("\n=== WIN/TIE/LOSS SUMMARY ===")
    generate_summary_stats(
        results,
        output_path=os.path.join(args.output_dir, 'summary.json')
    )
