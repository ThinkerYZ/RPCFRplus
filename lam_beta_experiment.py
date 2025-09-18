# file: lam_beta_experiment.py

import csv
import os
import sys

import numpy as np
from absl import app
from absl import flags
import pyspiel
from algorithm.RPDCFR_plus import RPDCFRPlusSolver
from algorithm.RPCFR_plus import RPCFRPlusSolver
from open_spiel.python.algorithms import exploitability

FLAGS = flags.FLAGS
algo = "rpcfr"
game = "battleship"
flags.DEFINE_integer("iterations", 2000, "Number of iterations")
# kuhn_poker, leduc_poker, liars_dice, goofspiel, battleship
flags.DEFINE_string("game", f"{game}", "Name of the game")
# rpcfr+, rpdcfr+

flags.DEFINE_string("algo", f"{algo}+", "Name of the game")
flags.DEFINE_string("results_dir", f"{algo}_{game}_lam_beta_experiments", "Directory to save results")
flags.DEFINE_float("alpha", 2.0, "Alpha parameter for weights")
flags.DEFINE_float("gamma", 5.0, "Gamma parameter for linear averaging")


def get_game(game_name):
    """Load game based on game name"""
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
    """Run a single experiment"""
    # 创建求解器
    if FLAGS.algo == "rpcfr+":
        solver = RPCFRPlusSolver(game, lam=lam, beta=beta, gamma=gamma)
    elif FLAGS.algo == "rpdcfr+":
            solver = RPDCFRPlusSolver(game, lam=lam, beta=beta, gamma=gamma)
    # solver = RPCFRPlusSolver(game, lam=lam, beta=beta, gamma=gamma)#, alpha=alpha)

    # Record the availability of each iteration
    exploitabilities = []

    for i in range(iterations):
        solver.evaluate_and_update_policy()
        if i % 10 == 0:  # Calculate availability every 10 iterations to save time
            conv = exploitability.exploitability(game, solver.average_policy())
            exploitabilities.append((i, conv))
            print(f"lam={lam:.3f}, beta={beta:.3f}, Iteration {i}: exploitability={conv}")

    # Return the average availability of the last few iterations
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


def run_all_combinations(game, results_dir, game_name, iterations, alpha, gamma):

    print("Running experiments for all combinations of lam and beta...")

    lam_values = [0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.99]
    beta_values = [0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.99]

    all_results = []

    total_combinations = len(lam_values) * len(beta_values)
    current_combination = 0

    for lam in lam_values:
        for beta in beta_values:
            current_combination += 1
            print(f"Running combination {current_combination}/{total_combinations}: lam={lam:.3f}, beta={beta:.3f}")

            # Run the experiment
            exploitabilities, final_exploitability = run_single_experiment(
                game, lam, beta, iterations, alpha, gamma
            )

            # Save results
            result = {
                'lam': lam,
                'beta': beta,
                'gamma': gamma,
                'alpha': alpha,
                'final_exploitability': final_exploitability
            }
            all_results.append(result)

            # Save detailed results
            save_experiment_result(results_dir, game_name, lam, beta, gamma, alpha, exploitabilities)

            print(f"Completed combination {current_combination}/{total_combinations}: "
                  f"lam={lam:.3f}, beta={beta:.3f}, final exploitability={final_exploitability:.6f}")

    # Save summary results
    summary_filename = f"{results_dir}/{game_name}_all_combinations_summary.csv"
    with open(summary_filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['lam', 'beta', 'gamma', 'alpha', 'final_exploitability'])
        for result in all_results:
            writer.writerow([result['lam'], result['beta'], result['gamma'], result['alpha'],
                             result['final_exploitability']])

    # Find the best results
    best_result = min(all_results, key=lambda x: x['final_exploitability'])
    print(f"\nBest result:")
    print(f"  lam: {best_result['lam']:.3f}")
    print(f"  beta: {best_result['beta']:.3f}")
    print(f"  gamma: {best_result['gamma']:.3f}")
    print(f"  alpha: {best_result['alpha']:.3f}")
    print(f"  final exploitability: {best_result['final_exploitability']:.6f}")

    return all_results, best_result


def main(_):

    os.makedirs(FLAGS.results_dir, exist_ok=True)
    game = get_game(FLAGS.game)
    all_results, best_result = run_all_combinations(
        game, FLAGS.results_dir, FLAGS.game, FLAGS.iterations, FLAGS.alpha, FLAGS.gamma
    )

    print(f"\nCompleted all experiments for {FLAGS.game}")
    print(f"Total combinations tested: {len(all_results)}")
    print(f"Best parameters found:")
    print(f"  lam: {best_result['lam']:.3f}")
    print(f"  beta: {best_result['beta']:.3f}")
    print(f"  final exploitability: {best_result['final_exploitability']:.6f}")


if __name__ == "__main__":
    app.run(main)
