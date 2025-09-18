import numpy as np

from utils import infoset_key, regret_matching, to_tabular_policy
from open_spiel.python.algorithms import exploitability

class TabularCFR:
    def __init__(self, game, iterations=10000):
        self.game = game
        self.iterations = iterations
        # Each player has an infoset dictionary：{info_key: {"regret_sum", "strategy_sum", "actions"}}
        self.infosets = [dict() for _ in range(game.num_players())]

    def _ensure_node(self, state, player):
        key = infoset_key(state, player)
        la = state.legal_actions()
        if key not in self.infosets[player]:
            self.infosets[player][key] = {
                "regret_sum": np.zeros(len(la), dtype=np.float64),
                "strategy_sum": np.zeros(len(la), dtype=np.float64),
                "actions": list(la)
            }
        return key, self.infosets[player][key]

    def _current_strategy(self, node):
        return regret_matching(node["regret_sum"])

    def _cfr(self, state, reach):
        """
        Reach=[p0.reach, p1_deach]: The CF probability of each reaching the information set (used by the other to update regrets)
        Return: The expected return for each player in that state (here is a zero sum, simply return a scalar)
        """
        if state.is_terminal():
            r = state.returns()  # List: Revenue per player
            return np.array(r, dtype=np.float64)

        if state.is_chance_node():
            util = np.zeros(self.game.num_players(), dtype=np.float64)
            for a, prob in state.chance_outcomes():
                child_util = self._cfr(state.child(a), reach)
                util += prob * child_util
            return util

        p = state.current_player()
        key, node = self._ensure_node(state, p)
        strategy = self._current_strategy(node)
        la = node["actions"]
        nA = len(la)

        utils = np.zeros((nA, self.game.num_players()), dtype=np.float64)
        node_util = np.zeros(self.game.num_players(), dtype=np.float64)

        for i, a in enumerate(la):
            new_reach = reach.copy()
            new_reach[p] *= strategy[i]
            child_util = self._cfr(state.child(a), new_reach)
            utils[i] = child_util
            node_util += strategy[i] * child_util

        # Only update the regret_Sum and strategy_sum of the current player
        cf_reach_opponent = reach[1 - p]
        # Instant regret vector (for current player)
        inst_regret = utils[:, p] - node_util[p]

        node["regret_sum"] += cf_reach_opponent * inst_regret
        node["strategy_sum"] += reach[p] * strategy

        return node_util

    def train(self, eval_every=1000):
        history = []
        for t in range(1, self.iterations + 1):
            self._cfr(self.game.new_initial_state(), [1.0, 1.0])
            if (t % eval_every) == 0:
                avg = self.average_strategy()
                tab = to_tabular_policy(self.game, avg)
                nc = exploitability.nash_conv(self.game, tab)
                history.append((t, nc))
        return history

    def average_strategy(self):
        out = [dict() for _ in range(self.game.num_players())]
        for p in range(self.game.num_players()):
            for key, node in self.infosets[p].items():
                strat_sum = node["strategy_sum"]
                z = strat_sum.sum()
                if z > 0:
                    out[p][key] = strat_sum / z
                else:
                    out[p][key] = np.ones_like(strat_sum) / len(strat_sum)
        return out
