"""
Visualization Module
====================
Generates plots matching figures in the C-TAEA paper:
  - 3D scatter plots of Pareto fronts (like Figs. 4-12)
  - Parallel Coordinate Plots for high-dimensional cases (like Figs. 13-56)
  - Box plots for HV comparison (like Fig. 15)
  - CV landscape plots (like Fig. 10)
"""

import numpy as np
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("Warning: matplotlib not available. Plotting disabled.")


# ---------------------------------------------------------------------------
# 3D Scatter Plot (like Fig. 4, 5, 6)
# ---------------------------------------------------------------------------

def scatter_3d(fronts_dict: dict, problem_name: str,
               output_path: str = None, show: bool = False):
    """
    Plot 3D scatter of Pareto fronts obtained by different algorithms.

    Parameters
    ----------
    fronts_dict : dict  { algo_name : np.ndarray (n, 3) }
    problem_name: str
    """
    if not HAS_MATPLOTLIB:
        return

    n_algos = len(fronts_dict)
    fig = plt.figure(figsize=(4 * n_algos, 4))

    for i, (algo_name, F) in enumerate(fronts_dict.items()):
        ax = fig.add_subplot(1, n_algos, i + 1, projection='3d')

        if len(F) > 0:
            ax.scatter(F[:, 0], F[:, 1], F[:, 2],
                       s=5, alpha=0.6, c='steelblue')
        else:
            ax.text(0.5, 0.5, 0.5, 'No feasible', ha='center', va='center',
                    transform=ax.transAxes)

        ax.set_xlabel('$f_1$', fontsize=8)
        ax.set_ylabel('$f_2$', fontsize=8)
        ax.set_zlabel('$f_3$', fontsize=8)
        ax.set_title(f'({chr(97+i)}) {algo_name}', fontsize=9)
        ax.tick_params(labelsize=6)

    fig.suptitle(f'{problem_name} — 3-objective', fontsize=11)
    plt.tight_layout()

    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved scatter plot: {output_path}")

    if show:
        plt.show()
    plt.close()


# ---------------------------------------------------------------------------
# Parallel Coordinate Plot (like Figs. 13-56)
# ---------------------------------------------------------------------------

def parallel_coords(fronts_dict: dict, problem_name: str, m: int,
                    output_path: str = None, show: bool = False):
    """
    Plot parallel coordinate plots for m-objective solutions.
    """
    if not HAS_MATPLOTLIB:
        return

    n_algos = len(fronts_dict)
    fig, axes = plt.subplots(1, n_algos, figsize=(3 * n_algos, 3))

    if n_algos == 1:
        axes = [axes]

    for i, (algo_name, F) in enumerate(fronts_dict.items()):
        ax = axes[i]

        if len(F) > 0:
            for row in F:
                ax.plot(range(1, m + 1), row, alpha=0.3, linewidth=0.5,
                        color='steelblue')

        ax.set_xlabel('Objective Index', fontsize=7)
        ax.set_ylabel('Objective Value', fontsize=7)
        ax.set_title(f'({chr(97+i)}) {algo_name}', fontsize=8)
        ax.set_xticks(range(1, m + 1))
        ax.tick_params(labelsize=6)

    fig.suptitle(f'{problem_name} — {m}-objective', fontsize=10)
    plt.tight_layout()

    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches='tight')

    if show:
        plt.show()
    plt.close()


# ---------------------------------------------------------------------------
# Box plots for HV (like Fig. 15)
# ---------------------------------------------------------------------------

def hv_boxplot(results_by_algo: dict, problem_name: str = '',
               output_path: str = None, show: bool = False):
    """
    Box plot of HV values across 51 runs.
    results_by_algo: { algo_name: [hv_values list] }
    """
    if not HAS_MATPLOTLIB:
        return

    fig, ax = plt.subplots(figsize=(8, 4))

    algo_names = list(results_by_algo.keys())
    data = [results_by_algo[a] for a in algo_names]

    bp = ax.boxplot(data, patch_artist=True,
                    medianprops=dict(color='black', linewidth=2))

    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
    for patch, color in zip(bp['boxes'], colors[:len(algo_names)]):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    ax.set_xticklabels(algo_names, fontsize=9)
    ax.set_ylabel('HV', fontsize=10)
    ax.set_title(f'Box plots of HV — {problem_name}', fontsize=11)
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()

    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved HV boxplot: {output_path}")

    if show:
        plt.show()
    plt.close()


# ---------------------------------------------------------------------------
# CV landscape plot (like Fig. 10)
# ---------------------------------------------------------------------------

def cv_landscape(problem_name: str, m: int,
                 output_path: str = None, show: bool = False):
    """
    Plot CV vs g(x_m) for DC2 problems (like Fig. 10 in paper).
    """
    if not HAS_MATPLOTLIB:
        return

    if problem_name not in ('DC2-DTLZ1', 'DC2-DTLZ3'):
        print(f"CV landscape only for DC2 problems, got {problem_name}")
        return

    configs = [
        {'a': 1, 'b': 0.5},
        {'a': 2, 'b': 0.5},
        {'a': 1, 'b': 0.9},
    ]
    g_vals = np.linspace(0, 10, 500)

    fig, axes = plt.subplots(1, 3, figsize=(12, 3))

    for idx, cfg in enumerate(configs):
        a, b = cfg['a'], cfg['b']
        ax = axes[idx]

        cv_vals = []
        for g in g_vals:
            c1 = np.cos(a * np.pi * g) - b
            c2 = np.exp(-g) - b
            cv = max(0.0, -c1) + max(0.0, -c2)
            cv_vals.append(cv)

        ax.plot(g_vals, cv_vals, 'b-', linewidth=1.5)
        ax.set_xlabel('$g(x_m)$', fontsize=10)
        ax.set_ylabel('$CV(x)$', fontsize=10)
        ax.set_title(f'a={a}, b={b}', fontsize=10)
        ax.grid(alpha=0.3)

    fig.suptitle(f'CV variation — {problem_name}', fontsize=11)
    plt.tight_layout()

    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches='tight')

    if show:
        plt.show()
    plt.close()


# ---------------------------------------------------------------------------
# Feasible region illustrations (2D, like Figs. 1-12 in supplement)
# ---------------------------------------------------------------------------

def plot_feasible_region_2d(problem_name: str, output_path: str = None,
                             show: bool = False):
    """
    Plot feasible region and PF in 2D objective space.
    Like the supplement figures 1-12.
    """
    if not HAS_MATPLOTLIB:
        return

    from Problems import ALL_PROBLEMS
    problem = ALL_PROBLEMS[problem_name](m=2)

    n_samples = 50000
    xl, xu = problem.xl, problem.xu
    X = np.random.uniform(xl, xu, (n_samples, problem.n_var))

    feasible_F = []
    infeasible_F = []

    for x in X:
        F, CV = problem.evaluate(x)
        if CV == 0.0:
            feasible_F.append(F)
        elif CV < 2.0:  # near-feasible
            infeasible_F.append(F)

    fig, ax = plt.subplots(figsize=(5, 5))

    if infeasible_F:
        inf_F = np.array(infeasible_F)
        ax.scatter(inf_F[:, 0], inf_F[:, 1], s=1, alpha=0.1,
                   color='lightgray', label='near-infeasible')

    if feasible_F:
        feas_F = np.array(feasible_F)
        ax.scatter(feas_F[:, 0], feas_F[:, 1], s=1, alpha=0.4,
                   color='steelblue', label='feasible')

    ax.set_xlabel('$f_1$', fontsize=12)
    ax.set_ylabel('$f_2$', fontsize=12)
    ax.set_title(f'{problem_name} — 2D feasible region', fontsize=11)
    ax.legend(markerscale=5, fontsize=9)
    ax.grid(alpha=0.3)

    plt.tight_layout()

    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches='tight')

    if show:
        plt.show()
    plt.close()


# ---------------------------------------------------------------------------
# Convenience: generate all figures from results
# ---------------------------------------------------------------------------

def generate_all_figures(results_dir: str = 'Results',
                          figures_dir: str = 'Analysis/figures',
                          quick: bool = False):
    """Generate all visualization figures from saved results."""
    import json

    os.makedirs(figures_dir, exist_ok=True)

    # CV landscape
    cv_landscape('DC2-DTLZ1', m=3,
                 output_path=os.path.join(figures_dir, 'fig10_cv_landscape.pdf'))

    # 2D feasible region illustrations
    for prob in ['C1-DTLZ1', 'C1-DTLZ3', 'C2-DTLZ2',
                 'C3-DTLZ1', 'C3-DTLZ4',
                 'DC1-DTLZ1', 'DC1-DTLZ3',
                 'DC2-DTLZ1', 'DC2-DTLZ3']:
        try:
            plot_feasible_region_2d(
                prob,
                output_path=os.path.join(figures_dir, f'feasible_{prob}.pdf')
            )
        except Exception as e:
            print(f"  Warning: could not plot {prob}: {e}")

    if quick:
        return

    print("Figures generation complete.")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--results_dir', default='Results')
    parser.add_argument('--figures_dir', default='Analysis/figures')
    parser.add_argument('--quick', action='store_true')
    args = parser.parse_args()

    generate_all_figures(args.results_dir, args.figures_dir, args.quick)
