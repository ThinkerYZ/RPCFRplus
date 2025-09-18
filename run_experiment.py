from pathlib import Path
import sys

from algorithm.pdcfr_plus.utils import run_method
from algorithm.pdcfr_plus.logger import Logger
from algorithm.pdcfr_plus.game_config import read_game_config

from algorithm.pdcfr_plus.pdcfr_plus import PDCFRPlus
from algorithm.pdcfr_plus.pcfr_plus import PCFRPlus

project_root = Path(__file__).absolute().parent
sys.path.insert(0, str(project_root))


def run_experiment(game_name, algo_name, iterations=2000, save_log=False,
                   alpha=None, beta=None, gamma=None, seed=0):

    # 构建配置字典
    configs = {
        "seed": seed,
        "algo_name": algo_name,
        "game_name": game_name,
        "iterations": iterations,
        "gamma": gamma,
        "alpha": alpha,
        "beta": beta,
        "save_log": save_log,
        "writer_strings": ["stdout"]
    }


    for arg in ["gamma", "alpha", "beta"]:
        if configs[arg] is None:
            del configs[arg]


    folder = f"results/{algo_name}"
    if save_log:
        configs["folder"] = folder


    game_config = run_method(read_game_config, configs)
    game_config.algo_name = algo_name
    print("game_config:", vars(game_config))
    if algo_name == "pdcfr+":
        solver = PDCFRPlus(game_config=game_config)
    elif algo_name == "pcfr+":
        # print('/////2////')
        solver = PCFRPlus(game_config=game_config)
    # print('/////3////')
    solver.learn()

