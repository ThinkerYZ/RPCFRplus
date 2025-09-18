# draw.py
import os
import csv
import matplotlib.pyplot as plt
import argparse
from collections import defaultdict


def load_data(file_path, max_iterations=None):
    """Load data from CSV file, with the option to only load data up to a specified number of iterations"""
    iterations = []
    exploitabilities = []

    with open(file_path, 'r') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            iteration = int(row['iteration'])
            # If the maximum number of iterations is specified and the current iteration exceeds that number, stop reading
            if max_iterations is not None and iteration > max_iterations:
                break
            iterations.append(iteration)
            exploitabilities.append(float(row['exploitability']))

    return iterations, exploitabilities


def find_available_games(results_dir="results"):
    """Scan the results directory to find all available game and algorithm combinations"""
    games_data = defaultdict(dict)  # {game_name: {algorithm: data_file_path}}

    if not os.path.exists(results_dir):
        print(f"Error: Cannot find directory {results_dir}")
        return games_data

    # Traverse algorithm directory
    for algorithm in os.listdir(results_dir):
        algorithm_path = os.path.join(results_dir, algorithm)
        if os.path.isdir(algorithm_path) and algorithm != "figures":
            # Traverse all game files in the algorithm directory
            for file in os.listdir(algorithm_path):
                if file.endswith(".csv"):
                    game_name = file[:-4]  # Remove the. csv suffix
                    file_path = os.path.join(algorithm_path, file)
                    games_data[game_name][algorithm] = file_path

    return games_data


def plot_game_comparison(game_name, algorithms_data, results_dir="results", max_iterations=None):
    """Draw a comparison chart of different algorithms in the same game"""
    plt.figure(figsize=(10, 6))

    # 修改后
    colors = [
        '#1f77b4',
        '#ff7f0e',
        '#2ca02c',
        '#9467bd',
        '#8c564b',
        '#e377c2',
        '#7f7f7f',
        '#bcbd22',
        '#17becf' ,
        '#d62728',
    ]

    has_data = False

    for i, (algorithm, file_path) in enumerate(algorithms_data.items()):
        if os.path.exists(file_path):
            try:
                iterations, exploitabilities = load_data(file_path, max_iterations)
                if iterations:  # Ensure that there is data to be plotted
                    plt.plot(iterations, exploitabilities,
                             label=algorithm.upper(),
                             color=colors[i % len(colors)],
                             linewidth=2)
                    has_data = True
            except Exception as e:
                print(f"Warning: Error reading file {file_path}: {e}")
        else:
            print(f"Warning: File not found{file_path}")

    if not has_data:
        print(f"Warning: Game {game_name} has no available data")
        plt.close()
        return

    plt.xlabel('Iteration')
    plt.ylabel('Exploitability')

    if max_iterations is not None:
        plt.title(f'Comparison of Algorithms on {game_name} (up to iteration {max_iterations})')
    else:
        plt.title(f'Comparison of Algorithms on {game_name}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.yscale('log')  # Use logarithmic scale to better display convergence status

    # save figure
    figures_dir = f"{results_dir}/figures"
    os.makedirs(figures_dir, exist_ok=True)
    if max_iterations is not None:
        filename = f"{figures_dir}/{game_name}_comparison_iterations_{max_iterations}.pdf"
    else:
        filename = f"{figures_dir}/{game_name}_comparison.pdf"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    # plt.show()
    plt.close()


def plot_all_games(games_data, results_dir="results", max_iterations=None):
    """Generate comparison charts for all games"""
    for game_name, algorithms_data in games_data.items():
        print(f"Generating comparison chart for game '{game_name}'...")
        plot_game_comparison(game_name, algorithms_data, results_dir, max_iterations)


def main():
    parser = argparse.ArgumentParser(description='Automatically read data from the results directory and draw algorithm comparison graphs')
    parser.add_argument('--results_dir', type=str, default='results', help='Result folder path')
    parser.add_argument('--max_iterations', type=int, default=2000,
                        help='Maximum number of iterations (only plot data up to this number of iterations)')

    args = parser.parse_args()

    # Search for all available game data
    games_data = find_available_games(args.results_dir)

    if not games_data:
        print("No game data was found. Please ensure that main.by is running to generate data.")
        return

    print(f"Find data for {len (games_data)} games:")
    for game_name, algorithms in games_data.items():
        print(f"  - {game_name}: {list(algorithms.keys())}")

    # If the maximum number of iterations is specified, display this information
    if args.max_iterations is not None:
        print(f"Limit the drawing of data to the {args. max_iterations} th iteration")

    # Generate charts for all games
    plot_all_games(games_data, args.results_dir, args.max_iterations)
    print("All charts have been generated!")


if __name__ == "__main__":
    main()
