from absl import app
from absl import flags
import csv
import os

from open_spiel.python.algorithms import cfr, discounted_cfr
from open_spiel.python.algorithms import exploitability
from algorithm.RPCFR_plus import RPCFRPlusSolver
from algorithm.RPDCFR_plus import RPDCFRPlusSolver
from run_experiment import run_experiment
import pyspiel

FLAGS = flags.FLAGS

flags.DEFINE_integer("iterations", 2000, "Number of iterations")
# kuhn_poker, leduc_poker, liars_dice, goofspiel, goofspielimp, battleship
flags.DEFINE_string("game", "leduc_poker", "Name of the game")
flags.DEFINE_enum("algorithm", "RPDCFR+",
                  ["cfr", "cfr+", "dcfr", "lcfr", "pcfr+", "pdcfr+", "RPCFR+",  "RPDCFR+"], "algorithm")
flags.DEFINE_integer("players", None, "Number of players")
flags.DEFINE_integer("print_freq", 1, "How often to print the exploitability")


def load_game(game_name):
    # if game_name == "goofspiel":
    #     game = pyspiel.load_game("goofspiel(num_cards=5,points_order=descending)")
    #     game = pyspiel.convert_to_turn_based(game)
    if game_name == "goofspiel":
        game = pyspiel.load_game("goofspiel(num_cards=5,points_order=descending)")
        game = pyspiel.convert_to_turn_based(game)
    elif game_name == "liars_dice":
        game = pyspiel.load_game("liars_dice(dice_sides=5,numdice=1,players=2)")
        # game = pyspiel.convert_to_turn_based(game)
    elif game_name == "battleship":
        game = pyspiel.load_game("battleship", {
            "allow_repeated_shots": False,
            "board_width": 2,
            "board_height": 2,
            "ship_sizes": "[2]",
            "ship_values": "[2]",
            "num_shots": 3,
        })
    elif game_name in ["kuhn_poker", "leduc_poker"]:
        game = pyspiel.load_game(FLAGS.game)
    elif game_name == "goofspielImp":
        game = pyspiel.load_game("goofspiel(num_cards=5,points_order=descending,imp_info=True)")
        game = pyspiel.convert_to_turn_based(game)
    # elif game_name == "chat_game":
    #     game = pyspiel.load_game("chat_game")
    #     game = pyspiel.convert_to_turn_based(game)
    else:
        raise ValueError(f"Unknown game: {game_name}")

    return game


def main(_):
  #---------------games----------
  games_list = pyspiel.registered_games()
  print("Registered games:")
  print(games_list)

  # for algorithm in ["cfr", "cfr+", "dcfr", "lcfr", "pcfr+", "pdcfr+", "RPCFR+",  "RPDCFR+"]:
  for algorithm in ["RPDCFR+"]:
      # for FLAGS.game in ["goofspielImp"]:
      for FLAGS.game in ["kuhn_poker", "leduc_poker", "liars_dice", "goofspiel",  "battleship"]:
          game = load_game(FLAGS.game )
          FLAGS.algorithm = algorithm
          if FLAGS.algorithm == "cfr":
              solver = cfr.CFRSolver(game)
          elif FLAGS.algorithm == "cfr+":
              solver = cfr.CFRPlusSolver(game)
          elif FLAGS.algorithm == "dcfr":
              solver = discounted_cfr.DCFRSolver(game)
          elif FLAGS.algorithm == "lcfr":
              solver = discounted_cfr.LCFRSolver(game)
          elif FLAGS.algorithm == "RPCFR+":
              solver = RPCFRPlusSolver(game)
          elif FLAGS.algorithm == "RPDCFR+":
              solver = RPDCFRPlusSolver(game)

          if FLAGS.algorithm in ["cfr", "cfr+", "dcfr", "lcfr", "RPCFR+", "RPDCFR+"]:
              # Create a folder named after the algorithm name
              algorithm_dir = f"results/{FLAGS.algorithm}"
              os.makedirs(algorithm_dir, exist_ok=True)
              csv_filename = f"{algorithm_dir}/{FLAGS.game}.csv"

              with open(csv_filename, 'w', newline='') as csvfile:
                  csv_writer = csv.writer(csvfile)
                  csv_writer.writerow(["iteration", "exploitability"])

                  for i in range(FLAGS.iterations):
                      solver.evaluate_and_update_policy()
                      if i % FLAGS.print_freq == 0:
                          conv = exploitability.exploitability(game, solver.average_policy())
                          print("Iteration {} exploitability {}".format(i, conv))
                          csv_writer.writerow([i, conv])

          elif FLAGS.algorithm in ["pcfr+", "pdcfr+"]:
              run_experiment(
                  game_name=FLAGS.game,
                  algo_name=FLAGS.algorithm,
                  iterations=FLAGS.iterations,
                  save_log=True,
                  alpha=1.5,
                  gamma=5
              )


if __name__ == "__main__":
  app.run(main)
