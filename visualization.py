# file: visualization.py

import csv
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd


def load_random_search_results(results_dir, game_name):

    filename = f"{results_dir}/{game_name}_random_search_summary.csv"
    if not os.path.exists(filename):
        return None

    data = pd.read_csv(filename)
    return data


def load_bayesian_optimization_results(results_dir, game_name):

    filename = f"{results_dir}/{game_name}_bayesian_optimization_details.csv"
    if not os.path.exists(filename):
        return None

    data = pd.read_csv(filename)
    return data


def plot_convergence_curve(results_dir, game_name):

    random_data = load_random_search_results(results_dir, game_name)
    bayesian_data = load_bayesian_optimization_results(results_dir, game_name)

    plt.rcParams.update({'font.size': 20})

    plt.figure(figsize=(12, 8))

    if random_data is not None:
        sorted_random = random_data.sort_values('trial')
        sorted_random['best_so_far'] = sorted_random['final_exploitability'].cummin()
        plt.plot(sorted_random['trial'], sorted_random['best_so_far'],
                 'o-', label='Random Search Best So Far', markersize=4)
        plt.plot(sorted_random['trial'], sorted_random['final_exploitability'],
                 'o', alpha=0.5, label='Random Search All Points', markersize=3)

    if bayesian_data is not None:
        bayesian_data['best_so_far'] = bayesian_data['exploitability'].cummin()
        random_trials = len(random_data) if random_data is not None else 0
        trial_indices = range(random_trials, random_trials + len(bayesian_data))
        plt.plot(trial_indices, bayesian_data['best_so_far'],
                 's-', label='Bayesian Optimization Best So Far', markersize=4)
        plt.plot(trial_indices, bayesian_data['exploitability'],
                 's', alpha=0.5, label='Bayesian Optimization All Points', markersize=3)

    plt.xlabel('Trial Number')
    plt.ylabel('Exploitability (Lower is Better)')
    plt.title(f'Convergence Curve for {game_name}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.yscale('log')  # 使用对数刻度更好地显示差异
    plt.tight_layout()
    plt.savefig(f"{results_dir}/{game_name}_convergence_curve.png", dpi=300, bbox_inches='tight')
    plt.close()


def plot_parameter_importance(results_dir, game_name):
    all_data = []

    random_data = load_random_search_results(results_dir, game_name)
    if random_data is not None:
        all_data.append(random_data)

    bayesian_data = load_bayesian_optimization_results(results_dir, game_name)
    if bayesian_data is not None:
        bayesian_data_renamed = bayesian_data.rename(columns={
            'exploitability': 'final_exploitability',
            'lam': 'lam',
            'beta': 'beta',
            'gamma': 'gamma',
            'alpha': 'alpha'
        })
        all_data.append(bayesian_data_renamed)

    if not all_data:
        return

    combined_data = pd.concat(all_data, ignore_index=True)

    plt.rcParams.update({'font.size': 20})

    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle(f'Parameter vs Exploitability for {game_name}', fontsize=24)

    parameters = ['lam', 'beta', 'gamma', 'alpha']
    axes = axes.flatten()

    for i, param in enumerate(parameters):
        ax = axes[i]

        jitter = np.random.normal(0, 0.01, len(combined_data))
        ax.scatter(combined_data[param] + jitter, combined_data['final_exploitability'],
                   alpha=0.6, s=20)
        ax.set_xlabel(param, fontsize=18)
        ax.set_ylabel('Exploitability', fontsize=18)
        ax.set_yscale('log')
        ax.grid(True, alpha=0.3)

        corr = combined_data[param].corr(combined_data['final_exploitability'])
        ax.set_title(f'{param} (correlation: {corr:.3f})', fontsize=18)

    plt.tight_layout()
    plt.savefig(f"{results_dir}/{game_name}_parameter_importance.png", dpi=300, bbox_inches='tight')
    plt.close()


def plot_parameter_heatmaps(results_dir, game_name):

    all_data = []

    random_data = load_random_search_results(results_dir, game_name)
    if random_data is not None:
        all_data.append(random_data)

    bayesian_data = load_bayesian_optimization_results(results_dir, game_name)
    if bayesian_data is not None:
        bayesian_data_renamed = bayesian_data.rename(columns={
            'exploitability': 'final_exploitability'
        })
        all_data.append(bayesian_data_renamed)

    if not all_data:
        return

    combined_data = pd.concat(all_data, ignore_index=True)

    plt.rcParams.update({'font.size': 20})


    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle(f'Parameter Pair Heatmaps for {game_name}', fontsize=24)

    param_pairs = [('lam', 'beta'), ('lam', 'gamma'), ('lam', 'alpha'),
                   ('beta', 'gamma'), ('beta', 'alpha'), ('gamma', 'alpha')]
    axes = axes.flatten()

    for i, (param1, param2) in enumerate(param_pairs):
        ax = axes[i]

        heatmap = ax.hist2d(combined_data[param1], combined_data[param2],
                            bins=20, cmap='viridis',
                            weights=1 / combined_data['final_exploitability'])
        ax.set_xlabel(param1, fontsize=18)
        ax.set_ylabel(param2, fontsize=18)
        cbar = plt.colorbar(heatmap[3], ax=ax)
        cbar.ax.tick_params(labelsize=16)
        ax.set_title(f'{param1} vs {param2}', fontsize=18)

    plt.tight_layout()
    plt.savefig(f"{results_dir}/{game_name}_parameter_heatmaps.png", dpi=300, bbox_inches='tight')
    plt.close()


def plot_3d_parameter_space(results_dir, game_name):

    all_data = []

    random_data = load_random_search_results(results_dir, game_name)
    if random_data is not None:
        all_data.append(random_data)

    bayesian_data = load_bayesian_optimization_results(results_dir, game_name)
    if bayesian_data is not None:
        bayesian_data_renamed = bayesian_data.rename(columns={
            'exploitability': 'final_exploitability'
        })
        all_data.append(bayesian_data_renamed)

    if not all_data:
        return

    combined_data = pd.concat(all_data, ignore_index=True)

    plt.rcParams.update({'font.size': 20})

    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')


    scatter = ax.scatter(combined_data['lam'], combined_data['beta'], combined_data['gamma'],
                         c=combined_data['final_exploitability'],
                         s=50 / np.log(combined_data['final_exploitability'] + 1),
                         cmap='viridis', alpha=0.7)

    ax.set_xlabel('Lambda', fontsize=18)
    ax.set_ylabel('Beta', fontsize=18)
    ax.set_zlabel('Gamma', fontsize=18)
    ax.set_title(f'3D Parameter Space for {game_name}', fontsize=20)

    cbar = plt.colorbar(scatter, ax=ax, shrink=0.5, aspect=5)
    cbar.ax.tick_params(labelsize=16)
    plt.tight_layout()
    plt.savefig(f"{results_dir}/{game_name}_3d_parameter_space.png", dpi=300, bbox_inches='tight')
    plt.close()


def plot_performance_distribution(results_dir, game_name):
    all_data = []

    random_data = load_random_search_results(results_dir, game_name)
    if random_data is not None:
        all_data.append(random_data)

    bayesian_data = load_bayesian_optimization_results(results_dir, game_name)
    if bayesian_data is not None:
        bayesian_data_renamed = bayesian_data.rename(columns={
            'exploitability': 'final_exploitability'
        })
        all_data.append(bayesian_data_renamed)

    if not all_data:
        return

    combined_data = pd.concat(all_data, ignore_index=True)

    plt.rcParams.update({'font.size': 20})

    plt.figure(figsize=(12, 8))

    plt.subplot(2, 2, 1)
    plt.hist(combined_data['final_exploitability'], bins=30, alpha=0.7, color='skyblue')
    plt.xlabel('Exploitability', fontsize=18)
    plt.ylabel('Frequency', fontsize=18)
    plt.title('Distribution of Exploitability Values', fontsize=18)
    plt.grid(True, alpha=0.3)

    plt.subplot(2, 2, 2)
    plt.hist(combined_data['final_exploitability'],
             bins=np.logspace(np.log10(combined_data['final_exploitability'].min()),
                              np.log10(combined_data['final_exploitability'].max()), 30),
             alpha=0.7, color='lightcoral')
    plt.xlabel('Exploitability', fontsize=18)
    plt.ylabel('Frequency', fontsize=18)
    plt.title('Distribution of Exploitability (Log Scale)', fontsize=18)
    plt.xscale('log')
    plt.grid(True, alpha=0.3)

    plt.subplot(2, 2, 3)
    methods = []
    values = []

    if random_data is not None:
        methods.extend(['Random Search'] * len(random_data))
        values.extend(random_data['final_exploitability'])

    if bayesian_data is not None:
        methods.extend(['Bayesian Opt'] * len(bayesian_data))
        values.extend(bayesian_data['exploitability'])

    df_box = pd.DataFrame({'Method': methods, 'Exploitability': values})
    sns.boxplot(data=df_box, x='Method', y='Exploitability')
    plt.title('Exploitability by Method', fontsize=18)
    plt.yscale('log')
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)

    plt.subplot(2, 2, 4)
    sns.violinplot(data=df_box, x='Method', y='Exploitability')
    plt.title('Exploitability Distribution by Method', fontsize=18)
    plt.yscale('log')
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)

    plt.tight_layout()
    plt.savefig(f"{results_dir}/{game_name}_performance_distribution.png", dpi=300, bbox_inches='tight')
    plt.close()


def create_summary_report(results_dir, game_name):

    all_data = []

    random_data = load_random_search_results(results_dir, game_name)
    if random_data is not None:
        all_data.append(random_data)

    bayesian_data = load_bayesian_optimization_results(results_dir, game_name)
    if bayesian_data is not None:
        bayesian_data_renamed = bayesian_data.rename(columns={
            'exploitability': 'final_exploitability'
        })
        all_data.append(bayesian_data_renamed)

    if not all_data:
        return

    combined_data = pd.concat(all_data, ignore_index=True)

    summary_stats = combined_data[['lam', 'beta', 'gamma', 'alpha', 'final_exploitability']].describe()

    best_idx = combined_data['final_exploitability'].idxmin()
    best_params = combined_data.iloc[best_idx]

    with open(f"{results_dir}/{game_name}_visualization_report.txt", 'w') as f:
        f.write(f"Visualization Report for {game_name}\n")
        f.write("=" * 50 + "\n\n")

        f.write("Summary Statistics:\n")
        f.write(summary_stats.to_string())
        f.write("\n\n")

        f.write("Best Parameters Found:\n")
        f.write(f"  Lambda: {best_params['lam']:.4f}\n")
        f.write(f"  Beta: {best_params['beta']:.4f}\n")
        f.write(f"  Gamma: {best_params['gamma']:.4f}\n")
        f.write(f"  Alpha: {best_params['alpha']:.4f}\n")
        f.write(f"  Exploitability: {best_params['final_exploitability']:.6f}\n\n")

        correlations = combined_data[['lam', 'beta', 'gamma', 'alpha', 'final_exploitability']].corr()[
                           'final_exploitability'][:-1]
        f.write("Parameter Correlations with Exploitability:\n")
        for param, corr in correlations.items():
            f.write(f"  {param}: {corr:.4f}\n")


def generate_all_visualizations(results_dir, game_name):

    print(f"Generating visualizations for {game_name}...")

    os.makedirs(results_dir, exist_ok=True)

    plt.style.use('seaborn-v0_8')
    sns.set_palette("husl")

    sns.set_context("paper", font_scale=2.0)

    try:
        plot_convergence_curve(results_dir, game_name)
        print("  - Convergence curve generated")
    except Exception as e:
        print(f"  - Error generating convergence curve: {e}")

    try:
        plot_parameter_importance(results_dir, game_name)
        print("  - Parameter importance plots generated")
    except Exception as e:
        print(f"  - Error generating parameter importance plots: {e}")

    try:
        plot_parameter_heatmaps(results_dir, game_name)
        print("  - Parameter heatmaps generated")
    except Exception as e:
        print(f"  - Error generating parameter heatmaps: {e}")

    try:
        plot_3d_parameter_space(results_dir, game_name)
        print("  - 3D parameter space plot generated")
    except Exception as e:
        print(f"  - Error generating 3D parameter space plot: {e}")

    try:
        plot_performance_distribution(results_dir, game_name)
        print("  - Performance distribution plots generated")
    except Exception as e:
        print(f"  - Error generating performance distribution plots: {e}")

    try:
        create_summary_report(results_dir, game_name)
        print("  - Summary report generated")
    except Exception as e:
        print(f"  - Error generating summary report: {e}")

    print(f"All visualizations saved to {results_dir}")


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 2:
        results_dir = sys.argv[1]
        game_name = sys.argv[2]
        generate_all_visualizations(results_dir, game_name)
    else:
        print("Usage: python visualization.py <results_dir> <game_name>")

