# file: run_opdcfr_plus_experiments_optimized.py

import csv
import os
import numpy as np
from absl import app
from absl import flags
import pyspiel
from algorithm.RPDCFR_plus import RPDCFRPlusSolver
from open_spiel.python.algorithms import exploitability

from visualization import generate_all_visualizations


try:
    from skopt import gp_minimize
    from skopt.space import Real
    from skopt.utils import use_named_args

    SKOPT_AVAILABLE = True
except ImportError:
    SKOPT_AVAILABLE = False
    print("skopt not available. Please install scikit-optimize for Bayesian optimization.")

FLAGS = flags.FLAGS
flags.DEFINE_integer("iterations", 2000, "Number of iterations")
# kuhn_poker, leduc_poker, liars_dice, goofspiel, goofspielimp, battleship
flags.DEFINE_string("game", "leduc_poker", "Name of the game")
flags.DEFINE_string("results_dir", "rpdcfr_plus_lam_beta_gamma_alpha", "Directory to save results")
flags.DEFINE_float("alpha", 2.0, "Alpha parameter for weights")
flags.DEFINE_float("gamma", 5.0, "Gamma parameter for linear averaging")
flags.DEFINE_integer("random_search_trials", 100, "Number of random search trials")
flags.DEFINE_integer("bayesian_opt_trials", 30, "Number of Bayesian optimization trials")
flags.DEFINE_bool("use_bayesian_opt", True, "Whether to use Bayesian optimization after random search")


def get_game(game_name):
    if game_name == "kuhn_poker":
        return pyspiel.load_game("kuhn_poker")
    elif game_name == "leduc_poker":
        return pyspiel.load_game("leduc_poker")
    elif game_name == "liars_dice":
        return pyspiel.load_game("liars_dice(dice_sides=5,numdice=1,players=2)")
    elif game_name == "goofspiel":
        game = pyspiel.load_game("goofspiel(num_cards=5,points_order=descending)")
        return pyspiel.convert_to_turn_based(game)
    elif game_name == "battleship":
        game = pyspiel.load_game("battleship", {
            "allow_repeated_shots": False,
            "board_width": 2,
            "board_height": 2,
            "ship_sizes": "[2]",
            "ship_values": "[2]",
            "num_shots": 3,
        })
        return game
    else:
        raise ValueError(f"Unknown game: {game_name}")


def run_single_experiment(game, lam, beta, iterations, alpha, gamma):

    solver = RPDCFRPlusSolver(game, lam=lam, beta=beta, gamma=gamma, alpha=alpha)
    exploitabilities = []
    for i in range(iterations):
        solver.evaluate_and_update_policy()
        if i % 10 == 0:
            conv = exploitability.exploitability(game, solver.average_policy())
            exploitabilities.append((i, conv))
            print(f"Iteration {i}: exploitability={conv}")

    final_exploitability = np.mean([exp for _, exp in exploitabilities[-10:]]) if len(exploitabilities) >= 10 else \
        exploitabilities[-1][1]

    return exploitabilities, final_exploitability


def save_experiment_result(results_dir, game_name, lam, beta, gamma, alpha, exploitabilities):
    filename = f"{results_dir}/{game_name}_lam_{lam:.3f}_beta_{beta:.3f}_gamma_{gamma:.3f}_alpha_{alpha:.3f}.csv"
    with open(filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['iteration', 'exploitability'])
        for iteration, exp in exploitabilities:
            writer.writerow([iteration, exp])
    return filename


def random_search(game, results_dir, game_name, iterations, n_trials=50):
    print(f"Starting random search with {n_trials} trials...")

    best_result = None
    best_exploitability = float('inf')
    all_results = []

    for trial in range(n_trials):
        lam_values = np.arange(0.0, 1.1, 0.1)
        beta_values = np.arange(0.0, 1.1, 0.1)
        gamma_values = np.arange(1, 11, 1)
        alpha_values = np.arange(1, 11, 1)

        lam = np.random.choice(lam_values)
        beta = np.random.choice(beta_values)
        gamma = np.random.choice(gamma_values)
        alpha = np.random.choice(alpha_values)
        print(
            f"Random search trial {trial + 1}/{n_trials}: lam={lam:.3f}, beta={beta:.3f}, gamma={gamma:.3f}, alpha={alpha:.3f}")
        exploitabilities, final_exploitability = run_single_experiment(
            game, lam, beta, iterations, alpha, gamma
        )

        result = {
            'trial': trial,
            'lam': lam,
            'beta': beta,
            'gamma': gamma,
            'alpha': alpha,
            'final_exploitability': final_exploitability
        }
        all_results.append(result)

        save_experiment_result(results_dir, game_name, lam, beta, gamma, alpha, exploitabilities)

        if final_exploitability < best_exploitability:
            best_exploitability = final_exploitability
            best_result = result.copy()
            print(f"New best exploitability: {final_exploitability:.6f}")

    summary_filename = f"{results_dir}/{game_name}_random_search_summary.csv"
    with open(summary_filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['trial', 'lam', 'beta', 'gamma', 'alpha', 'final_exploitability'])
        for result in all_results:
            writer.writerow([result['trial'], result['lam'], result['beta'], result['gamma'], result['alpha'],
                             result['final_exploitability']])

    return best_result, all_results


def bayesian_optimization(game, results_dir, game_name, iterations, n_trials=30, initial_points=None):
    if not SKOPT_AVAILABLE:
        print("Skipping Bayesian optimization as skopt is not available.")
        return None

    print(f"Starting Bayesian optimization with {n_trials} trials...")

    dimensions = [
        Real(0.0, 1.0, name='lam'),
        Real(0.0, 1.0, name='beta'),
        Real(1.0, 10.0, name='gamma'),
        Real(1.0, 10.0, name='alpha')
    ]

    initial_points_list = []
    if initial_points:
        for point in initial_points:
            lam = max(0.0, min(1.0, point['lam']))
            beta = max(0.0, min(1.0, point['beta']))
            gamma = max(1.0, min(10.0, point['gamma']))
            alpha = max(1.0, min(10.0, point['alpha']))
            initial_points_list.append([lam, beta, gamma, alpha])
            initial_points_list.append([point['lam'], point['beta'], point['gamma'], point['alpha']])
    @use_named_args(dimensions)
    def objective_function(lam, beta, gamma, alpha):
        print(f"Bayesian optimization trial: lam={lam:.3f}, beta={beta:.3f}, gamma={gamma:.3f}, alpha={alpha:.3f}")

        _, final_exploitability = run_single_experiment(
            game, lam, beta, iterations, alpha, gamma
        )

        return final_exploitability

    result = gp_minimize(
        func=objective_function,
        dimensions=dimensions,
        n_calls=n_trials,
        x0=initial_points_list if initial_points_list else None,
        random_state=42,
        n_random_starts=max(10, n_trials // 4)
    )

    bo_filename = f"{results_dir}/{game_name}_bayesian_optimization_details.csv"
    with open(bo_filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['trial', 'lam', 'beta', 'gamma', 'alpha', 'exploitability'])
        for i, (params, exp) in enumerate(zip(result.x_iters, result.func_vals)):
            writer.writerow([i + 1, params[0], params[1], params[2], params[3], exp])

    best_result = {
        'lam': result.x[0],
        'beta': result.x[1],
        'gamma': result.x[2],
        'alpha': result.x[3],
        'final_exploitability': result.fun
    }

    return best_result


def run_experiments():
    os.makedirs(FLAGS.results_dir, exist_ok=True)
    game = get_game(FLAGS.game)
    best_random_result, all_random_results = random_search(
        game, FLAGS.results_dir, FLAGS.game, FLAGS.iterations, FLAGS.random_search_trials
    )
    print(f"\nBest result from random search:")
    print(f"  lam: {best_random_result['lam']:.3f}")
    print(f"  beta: {best_random_result['beta']:.3f}")
    print(f"  gamma: {best_random_result['gamma']:.3f}")
    print(f"  alpha: {best_random_result['alpha']:.3f}")
    print(f"  final exploitability: {best_random_result['final_exploitability']:.6f}")

    best_overall_result = best_random_result
    if FLAGS.use_bayesian_opt and SKOPT_AVAILABLE:
        sorted_results = sorted(all_random_results, key=lambda x: x['final_exploitability'])
        top_initial_points = sorted_results[:min(5, len(sorted_results))]
        best_bayesian_result = bayesian_optimization(
            game, FLAGS.results_dir, FLAGS.game, FLAGS.iterations,
            FLAGS.bayesian_opt_trials, top_initial_points
        )

        if best_bayesian_result and best_bayesian_result['final_exploitability'] < best_random_result[
            'final_exploitability']:
            best_overall_result = best_bayesian_result
            print(f"\nBayesian optimization found better result:")
            print(f"  lam: {best_bayesian_result['lam']:.3f}")
            print(f"  beta: {best_bayesian_result['beta']:.3f}")
            print(f"  gamma: {best_bayesian_result['gamma']:.3f}")
            print(f"  alpha: {best_bayesian_result['alpha']:.3f}")
            print(f"  final exploitability: {best_bayesian_result['final_exploitability']:.6f}")
        else:
            print("\nBayesian optimization did not improve upon random search results.")

    print(f"\nFinal best parameters for {FLAGS.game}:")
    print(f"  lam: {best_overall_result['lam']:.3f}")
    print(f"  beta: {best_overall_result['beta']:.3f}")
    print(f"  gamma: {best_overall_result['gamma']:.3f}")
    print(f"  alpha: {best_overall_result['alpha']:.3f}")
    print(f"  final exploitability: {best_overall_result['final_exploitability']:.6f}")

    generate_all_visualizations(FLAGS.results_dir, FLAGS.game)


def load_existing_random_search_results(results_dir, game_name):

    summary_file = f"{results_dir}/{game_name}_random_search_summary.csv"
    if not os.path.exists(summary_file):
        print(f"No existing random search results found at {summary_file}")
        return None, []

    all_results = []
    best_result = None
    best_exploitability = float('inf')

    try:
        with open(summary_file, 'r') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                result = {
                    'trial': int(row['trial']),
                    'lam': float(row['lam']),
                    'beta': float(row['beta']),
                    'gamma': float(row['gamma']),
                    'alpha': float(row['alpha']),
                    'final_exploitability': float(row['final_exploitability'])
                }
                all_results.append(result)

                if result['final_exploitability'] < best_exploitability:
                    best_exploitability = result['final_exploitability']
                    best_result = result.copy()
        print(f"Loaded {len(all_results)} existing random search results")
        return best_result, all_results
    except Exception as e:
        print(f"Error loading existing random search results: {e}")
        return None, []


def load_run_experiments():

    os.makedirs(FLAGS.results_dir, exist_ok=True)
    game = get_game(FLAGS.game)
    existing_best_result, existing_all_results = load_existing_random_search_results(
        FLAGS.results_dir, FLAGS.game
    )

    if existing_best_result is not None and len(existing_all_results) >= 100:
        print(f"Using {len(existing_all_results)} existing random search results")
        best_random_result = existing_best_result
        all_random_results = existing_all_results
    else:
        best_random_result, all_random_results = random_search(
            game, FLAGS.results_dir, FLAGS.game, FLAGS.iterations, FLAGS.random_search_trials
        )

    print(f"\nBest result from random search:")
    print(f"  lam: {best_random_result['lam']:.3f}")
    print(f"  beta: {best_random_result['beta']:.3f}")
    print(f"  gamma: {best_random_result['gamma']:.3f}")
    print(f"  alpha: {best_random_result['alpha']:.3f}")
    print(f"  final exploitability: {best_random_result['final_exploitability']:.6f}")

    best_overall_result = best_random_result
    if FLAGS.use_bayesian_opt and SKOPT_AVAILABLE:
        sorted_results = sorted(all_random_results, key=lambda x: x['final_exploitability'])
        top_initial_points = sorted_results[:min(5, len(sorted_results))]

        best_bayesian_result = bayesian_optimization(
            game, FLAGS.results_dir, FLAGS.game, FLAGS.iterations,
            FLAGS.bayesian_opt_trials, top_initial_points
        )

        if best_bayesian_result and best_bayesian_result['final_exploitability'] < best_random_result[
            'final_exploitability']:
            best_overall_result = best_bayesian_result
            print(f"\nBayesian optimization found better result:")
            print(f"  lam: {best_bayesian_result['lam']:.3f}")
            print(f"  beta: {best_bayesian_result['beta']:.3f}")
            print(f"  gamma: {best_bayesian_result['gamma']:.3f}")
            print(f"  alpha: {best_bayesian_result['alpha']:.3f}")
            print(f"  final exploitability: {best_bayesian_result['final_exploitability']:.6f}")
        else:
            print("\nBayesian optimization did not improve upon random search results.")

    print(f"\nFinal best parameters for {FLAGS.game}:")
    print(f"  lam: {best_overall_result['lam']:.3f}")
    print(f"  beta: {best_overall_result['beta']:.3f}")
    print(f"  gamma: {best_overall_result['gamma']:.3f}")
    print(f"  alpha: {best_overall_result['alpha']:.3f}")
    print(f"  final exploitability: {best_overall_result['final_exploitability']:.6f}")

    generate_all_visualizations(FLAGS.results_dir, FLAGS.game)


def main(_):
    # run_experiments()
    load_run_experiments()


if __name__ == "__main__":
    app.run(main)
