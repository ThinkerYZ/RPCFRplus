import numpy as np
from .CFR import _CFRSolverBase
import collections


class RPDCFRPlusSolver(_CFRSolverBase):
    def __init__(self, game, lam=0.6, beta=0.3, gamma=5, alpha=3):
        super().__init__(
            game,
            regret_matching_plus=True,
            alternating_updates=True,
            linear_averaging=True,
        )
        self._lam = lam      # Prediction strength: 0~1, the larger it is, the more dependent it is on the prediction signal
        self._beta = beta    # EMA coefficient: 0~1, the larger the value, the more "up-to-date" it is
        self._gamma = gamma  # Strategy average weighted (maintaining long-term stability)
        self._alpha = alpha

    def weight(self, T, alpha):
        d, w = np.power(T - 1, alpha) / (np.power(T - 1, alpha) + 1), 1
        return d, w

    def evaluate_and_update_policy(self):
        self._iteration += 1
        for player in range(self._game.num_players()):
            self._compute_counterfactual_regret_for_player(
                self._root_node,
                policies=None,
                reach_probabilities=np.ones(self._game.num_players() + 1),
                player=player,
            )

            # def update_regret(self, T, alpha):
            #     d, w = self.weight(T, alpha)
            #     for a in self.legal_actions:
            #         self.regrets[a] = max(self.regrets[a] * d + self.imm_regrets[a] * w, 0)
            d, w = self.weight(self._iteration, self._alpha)
            for node in self._info_state_nodes.values():
                for a in node.legal_actions:
                    node.cumulative_regret[a] = max(
                        node.cumulative_regret[a] * d + node.imm_regret[a] * w, 0
                    )

            self._update_current_policy_with_prediction()

            for node in self._info_state_nodes.values():
                node.imm_regret = collections.defaultdict(float)

    def _compute_counterfactual_regret_for_player(self, state, policies,
                                                  reach_probabilities, player):
        if state.is_terminal():
            return np.asarray(state.returns())

        if state.is_chance_node():
            state_value = 0.0
            for action, action_prob in state.chance_outcomes():
                new_state = state.child(action)
                new_reach_probabilities = reach_probabilities.copy()
                new_reach_probabilities[-1] *= action_prob
                state_value += action_prob * self._compute_counterfactual_regret_for_player(
                    new_state, policies, new_reach_probabilities, player)
            return state_value

        current_player = state.current_player()
        info_state = state.information_state_string(current_player)

        if all(reach_probabilities[:-1] == 0):
            return np.zeros(self._num_players)

        state_value = np.zeros(self._num_players)
        children_utilities = {}

        node = self._info_state_nodes[info_state]
        if not hasattr(node, 'imm_regret'):
            node.imm_regret = collections.defaultdict(float)

        if policies is None:
            info_state_policy = self._get_infostate_policy(info_state)
        else:
            info_state_policy = policies[current_player](info_state)

        for action in state.legal_actions():
            action_prob = info_state_policy.get(action, 0.0)
            new_state = state.child(action)
            new_reach_probabilities = reach_probabilities.copy()
            new_reach_probabilities[current_player] *= action_prob
            child_utility = self._compute_counterfactual_regret_for_player(
                new_state, policies, new_reach_probabilities, player
            )
            state_value += action_prob * child_utility
            children_utilities[action] = child_utility

        simulatenous_updates = player is None
        if not simulatenous_updates and current_player != player:
            return state_value

        reach_prob = reach_probabilities[current_player]
        counterfactual_reach_prob = (
            np.prod(reach_probabilities[:current_player])
            * np.prod(reach_probabilities[current_player + 1:])
        )
        state_value_for_player = state_value[current_player]

        for action, action_prob in info_state_policy.items():
            node.imm_regret.setdefault(action, 0.0)

            cfr_regret = counterfactual_reach_prob * (
                children_utilities[action][current_player] - state_value_for_player
            )
            node.imm_regret[action] += cfr_regret

            # Cumulative strategy (with gamma weight)
            node.cumulative_policy[action] = (
                node.cumulative_policy[action] * ((self._iteration - 1) / self._iteration) ** self._gamma
                + reach_prob * action_prob
            )

        return state_value

    def _update_current_policy_with_prediction(self):

        d, w = self.weight(self._iteration + 1, self._alpha)
        for info_state, node in self._info_state_nodes.items():

            if not hasattr(node, 'pred_signal'):
                node.pred_signal = collections.defaultdict(float)

            #    φ_t(a) = (1-β)*φ_{t-1}(a) + β*imm_regret_t(a)
            for a in node.legal_actions:
                imm = node.imm_regret.get(a, 0.0)
                node.pred_signal[a] = (1.0 - self._beta) * node.pred_signal[a] * d + self._beta * imm * w

            state_policy = self._current_policy.policy_for_key(info_state)

            pred_regrets = {}
            for a in node.legal_actions:
                R = node.cumulative_regret.get(a, 0.0)
                phi = node.pred_signal.get(a, 0.0)
                pred_regrets[a] = max((1.0 - self._lam) * R + self._lam * phi, 0.0)

            reg_sum = sum(pred_regrets.values())
            if reg_sum > 0:
                for a in node.legal_actions:
                    state_policy[a] = pred_regrets[a] / reg_sum
            else:
                uniform = 1.0 / len(node.legal_actions)
                for a in node.legal_actions:
                    state_policy[a] = uniform
