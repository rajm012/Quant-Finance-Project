

"""
Analysis module: generates LaTeX tables and statistics matching the paper.
Performs Wilcoxon rank-sum test for statistical comparison.
"""


import numpy as np
import json
from pathlib import Path
from scipy import stats


# ─────────────────────────────────────────────────────────────────────────────
# Statistical tests
# ─────────────────────────────────────────────────────────────────────────────

def _coerce_numeric(values):
    """Convert mixed JSON-loaded values into numeric floats."""
    out = []
    for v in values:
        if isinstance(v, str):
            lv = v.strip().lower()
            if lv in {'inf', '+inf', 'infinity', '+infinity'}:
                out.append(np.inf)
            elif lv in {'-inf', '-infinity'}:
                out.append(-np.inf)
            elif lv in {'nan'}:
                out.append(np.nan)
            else:
                out.append(float(v))
        else:
            out.append(float(v))
    return out


def _prepare_metric_values(values, metric):
    """Prepare values for statistics while preserving failure signal.

    For IGD (lower is better), +inf is mapped to a large finite penalty so
    failed runs remain represented in rank tests.
    """
    vals = _coerce_numeric(values)
    vals = [x for x in vals if not np.isnan(x)]
    if not vals:
        return []

    if metric == 'igd':
        finite = [x for x in vals if np.isfinite(x)]
        penalty = (max(finite) * 10.0) if finite else 1e12
        return [penalty if np.isposinf(x) else x for x in vals]

    # HV (higher is better): keep finite values; if inf appears, cap to a large
    # finite value to keep rank test stable.
    finite = [x for x in vals if np.isfinite(x)]
    if not finite:
        return []
    cap = max(finite) * 10.0 if finite else 1.0
    return [cap if np.isposinf(x) else x for x in vals if not np.isneginf(x)]


def wilcoxon_test(data1, data2, metric='igd', alpha=0.05):
    """
    Wilcoxon rank-sum test (Mann-Whitney U).
    Returns '+' if data1 significantly better, '-' if worse, '=' if no difference.
    Better = smaller IGD or larger HV.
    """
    
    data1 = _prepare_metric_values(data1, metric)
    data2 = _prepare_metric_values(data2, metric)

    if len(data1) < 3 or len(data2) < 3:
        return '='

    try:
        stat, p_val = stats.mannwhitneyu(data1, data2, alternative='two-sided')
        if p_val < alpha:
            if metric == 'igd':
                return '+' if np.median(data1) < np.median(data2) else '-'
            return '+' if np.median(data1) > np.median(data2) else '-'
        return '='
    
    except Exception:
        return '='
    


def compare_algorithms(results, metric='igd'):
    """
    Compare C-TAEA against all peer algorithms using Wilcoxon test.

    Parameters
    ----------
    results : dict  loaded from all_results.json
    metric  : 'igd' or 'hv'

    Returns
    -------
    comparison : dict  {prob: {m: {algo: symbol}}}
    """
    
    comparison = {}
    runs_key = 'runs'

    for prob in results:
        comparison[prob] = {}
        for m_str in results[prob]:
            m = int(m_str)
            comparison[prob][m] = {}
            algos = results[prob][m_str]
            if 'C-TAEA' not in algos:
                continue

            ctaea_vals = [r.get(metric, np.inf) for r in algos['C-TAEA'].get(runs_key, [])]

            for algo_name, algo_data in algos.items():
                if algo_name == 'C-TAEA':
                    continue
                
                peer_vals = [r.get(metric, np.inf) for r in algo_data.get(runs_key, [])]
                
                if metric == 'igd':
                    
                    # Lower is better for IGD
                    sym = wilcoxon_test(ctaea_vals, peer_vals, metric='igd')
                    
                    # '+' means C-TAEA better (lower), shown as '†' in paper
                    comparison[prob][m][algo_name] = '†' if sym == '+' else ('‡' if sym == '-' else '')
                    
                else:
                    # Higher is better for HV
                    sym = wilcoxon_test(ctaea_vals, peer_vals, metric='hv')
                    comparison[prob][m][algo_name] = '†' if sym == '+' else ('‡' if sym == '-' else '')

    return comparison


# ─────────────────────────────────────────────────────────────────────────────
# Table generation
# ─────────────────────────────────────────────────────────────────────────────

def format_sci(val, iqr=None):
    """Format a value in scientific notation like the paper."""
    if np.isinf(val) or np.isnan(val):
        return r'\infty'
    s = f'{val:.3E}'
    
    # Convert to LaTeX-friendly format
    mantissa, exp = s.split('E')
    exp_int = int(exp)
    
    if iqr is not None and not (np.isinf(iqr) or np.isnan(iqr)):
        iqr_s = f'{iqr:.2E}'
        iqr_m, iqr_e = iqr_s.split('E')
        return f'{float(mantissa):.3f}e{exp_int:+d}({float(iqr_m):.2f}e{int(iqr_e):+d})'
    
    return f'{float(mantissa):.3f}e{exp_int:+d}'


def generate_igd_table(results, problems, m_values, algo_names, metric='igd'):
    """
    Generate a summary table matching Table I/III from the paper.

    Parameters
    ----------
    results    : dict
    problems   : list of problem names
    m_values   : list of m values
    algo_names : list of algorithm names
    metric     : 'igd' or 'hv'
    """
    
    med_key = f'{metric}_median'
    iqr_key = f'{metric}_iqr'

    header = f"{'Problem':12s} {'m':3s}"
    for a in algo_names:
        header += f"  {a:>18s}"
        
    print(header)
    print('-' * len(header))

    for prob in problems:
        for m in m_values:
            m_str = str(m)
            if prob not in results or m_str not in results[prob]:
                continue

            row = f"{prob:12s} {m:3d}"
            algos_data = results[prob][m_str]

            # Find best value
            vals = {}
            for a in algo_names:
                if a in algos_data:
                    v = algos_data[a].get(med_key, np.inf)
                    vals[a] = v if v is not None else np.inf

            if metric == 'igd':
                best_val = min(vals.values()) if vals else np.inf
            else:
                best_val = max(vals.values()) if vals else -np.inf

            for a in algo_names:
                if a in algos_data:
                    v = algos_data[a].get(med_key, np.inf)
                    iqr = algos_data[a].get(iqr_key, 0.0)
                    if v is None:
                        v = np.inf
                    
                    if iqr is None:
                        iqr = 0.0

                    is_best = (metric == 'igd' and abs(v - best_val) < 1e-10) or (metric == 'hv' and abs(v - best_val) < 1e-10)
                    cell = format_sci(v, iqr)
                    
                    if is_best:
                        cell = f'*{cell}*'
                    
                    row += f"  {cell:>18s}"
                
                else:
                    row += f"  {'N/A':>18s}"

            print(row)
    print()



def generate_summary_wins(results, algo_names, metric='igd'):
    """
    Generate win/loss/tie summary.
    C-TAEA vs each peer algorithm.
    """
    wins = {a: 0 for a in algo_names if a != 'C-TAEA'}
    losses = {a: 0 for a in algo_names if a != 'C-TAEA'}
    ties = {a: 0 for a in algo_names if a != 'C-TAEA'}

    for prob in results:
        for m_str in results[prob]:
            algos = results[prob][m_str]
            if 'C-TAEA' not in algos:
                continue

            ctaea_vals = [r.get(metric, np.inf) for r in algos['C-TAEA'].get('runs', [])]
            for a in algo_names:
                if a == 'C-TAEA' or a not in algos:
                    continue
                
                peer_vals = [r.get(metric, np.inf) for r in algos[a].get('runs', [])]
                sym = wilcoxon_test(ctaea_vals, peer_vals, metric=metric)
                
                if metric == 'igd':
                    if sym == '+':
                        wins[a] += 1
                    elif sym == '-':
                        losses[a] += 1
                    else:
                        ties[a] += 1
                        
                else:
                    if sym == '+':
                        wins[a] += 1
                    elif sym == '-':
                        losses[a] += 1
                    else:
                        ties[a] += 1

    print(f"\nC-TAEA vs peers ({metric.upper()}):")
    print(f"{'Algorithm':20s} {'W':5s} {'L':5s} {'T':5s}")
    print('-' * 35)
    
    for a in algo_names:
        if a == 'C-TAEA':
            continue
        
        print(f"{a:20s} {wins[a]:5d} {losses[a]:5d} {ties[a]:5d}")



def load_and_analyze(results_dir='Results'):
    """
    Load results from JSON and generate paper-style analysis.
    """
    results_path = Path(results_dir) / 'all_results.json'
    if not results_path.exists():
        print(f"Results file not found: {results_path}")
        return None

    with open(results_path) as f:
        results = json.load(f)

    algo_names = ['C-TAEA', 'C-NSGA-III', 'C-MOEA/D', 'C-MOEA/DD', 'I-DBEA', 'CMOEA']
    cdtlz_problems = ['C1DTLZ1', 'C1DTLZ3', 'C2DTLZ2', 'C3DTLZ1', 'C3DTLZ4']
    dcdtlz_problems = ['DC1DTLZ1', 'DC1DTLZ3', 'DC2DTLZ1', 'DC2DTLZ3', 'DC3DTLZ1', 'DC3DTLZ3']
    m_values = [3, 5, 8, 10, 15]

    print("\n" + "="*80)
    print("TABLE I: C-DTLZ Benchmark Suite - IGD Results")
    print("="*80)
    
    generate_igd_table(results, cdtlz_problems, m_values, algo_names, 'igd')

    print("\n" + "="*80)
    print("TABLE III: DC-DTLZ Benchmark Suite - IGD Results")
    print("="*80)
    
    generate_igd_table(results, dcdtlz_problems, m_values, algo_names, 'igd')

    print("\n" + "="*80)
    print("HV Results - C-DTLZ")
    print("="*80)
    
    generate_igd_table(results, cdtlz_problems, m_values, algo_names, 'hv')

    print("\n" + "="*80)
    print("HV Results - DC-DTLZ")
    print("="*80)

    generate_igd_table(results, dcdtlz_problems, m_values, algo_names, 'hv')

    generate_summary_wins(results, algo_names, 'igd')
    generate_summary_wins(results, algo_names, 'hv')

    return results


