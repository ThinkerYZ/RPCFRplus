import collections
import attr
import numpy as np

from open_spiel.python import policy
import pyspiel


@attr.s
class _InfoStateNode(object):
  """An object wrapping values associated to an information state."""
  legal_actions = attr.ib()
  index_in_tabular_policy = attr.ib()
  cumulative_regret = attr.ib(factory=lambda: collections.defaultdict(float))
  cumulative_policy = attr.ib(factory=lambda: collections.defaultdict(float))
  predicted_regret = attr.ib(factory=lambda: collections.defaultdict(float))


def _apply_regret_matching_plus_reset(info_state_nodes):
  for info_state_node in info_state_nodes.values():
    action_to_cum_regret = info_state_node.cumulative_regret
    for action, cumulative_regret in action_to_cum_regret.items():
      if cumulative_regret < 0:
        action_to_cum_regret[action] = 0


def _update_current_policy(current_policy, info_state_nodes):
  for info_state, info_state_node in info_state_nodes.items():
    state_policy = current_policy.policy_for_key(info_state)

    for action, value in _regret_matching(
        info_state_node.cumulative_regret,
        info_state_node.legal_actions).items():
      state_policy[action] = value


def _update_average_policy(average_policy, info_state_nodes):
  for info_state, info_state_node in info_state_nodes.items():
    info_state_policies_sum = info_state_node.cumulative_policy
    state_policy = average_policy.policy_for_key(info_state)
    probabilities_sum = sum(info_state_policies_sum.values())
    if probabilities_sum == 0:
      num_actions = len(info_state_node.legal_actions)
      for action in info_state_node.legal_actions:
        state_policy[action] = 1 / num_actions
    else:
      for action, action_prob_sum in info_state_policies_sum.items():
        state_policy[action] = action_prob_sum / probabilities_sum


class _CFRSolverBase(object):
  def __init__(self, game, alternating_updates, linear_averaging,
               regret_matching_plus):
    # pyformat: enable
    assert game.get_type().dynamics == pyspiel.GameType.Dynamics.SEQUENTIAL, (
        "CFR requires sequential games. If you're trying to run it " +
        "on a simultaneous (or normal-form) game, please first transform it " +
        "using turn_based_simultaneous_game.")

    self._game = game
    self._num_players = game.num_players()
    self._root_node = self._game.new_initial_state()

    # This is for returning the current policy and average policy to a caller
    self._current_policy = policy.TabularPolicy(game)
    self._average_policy = self._current_policy.__copy__()

    self._info_state_nodes = {}
    self._initialize_info_state_nodes(self._root_node)

    self._iteration = 0  # For possible linear-averaging.
    self._linear_averaging = linear_averaging
    self._alternating_updates = alternating_updates
    self._regret_matching_plus = regret_matching_plus

  def _initialize_info_state_nodes(self, state):
    if state.is_terminal():
      return

    if state.is_chance_node():
      for action, unused_action_prob in state.chance_outcomes():
        self._initialize_info_state_nodes(state.child(action))
      return

    current_player = state.current_player()
    info_state = state.information_state_string(current_player)

    info_state_node = self._info_state_nodes.get(info_state)
    if info_state_node is None:
      legal_actions = state.legal_actions(current_player)
      info_state_node = _InfoStateNode(
          legal_actions=legal_actions,
          index_in_tabular_policy=self._current_policy.state_lookup[info_state])
      self._info_state_nodes[info_state] = info_state_node

    for action in info_state_node.legal_actions:
      self._initialize_info_state_nodes(state.child(action))

  def current_policy(self):
    return self._current_policy

  def average_policy(self):
    _update_average_policy(self._average_policy, self._info_state_nodes)
    return self._average_policy

  def _compute_counterfactual_regret_for_player(self, state, policies,
                                                reach_probabilities, player):
    if state.is_terminal():
      return np.asarray(state.returns())

    if state.is_chance_node():
      state_value = 0.0
      for action, action_prob in state.chance_outcomes():
        assert action_prob > 0
        new_state = state.child(action)
        new_reach_probabilities = reach_probabilities.copy()
        new_reach_probabilities[-1] *= action_prob
        state_value += action_prob * self._compute_counterfactual_regret_for_player(
            new_state, policies, new_reach_probabilities, player)
      return state_value

    current_player = state.current_player()
    info_state = state.information_state_string(current_player)

    # No need to continue on this history branch as no update will be performed
    # for any player.
    # The value we return here is not used in practice. If the conditional
    # statement is True, then the last taken action has probability 0 of
    # occurring, so the returned value is not impacting the parent node value.
    if all(reach_probabilities[:-1] == 0):
      return np.zeros(self._num_players)

    state_value = np.zeros(self._num_players)

    # The utilities of the children states are computed recursively. As the
    # regrets are added to the information state regrets for each state in that
    # information state, the recursive call can only be made once per child
    # state. Therefore, the utilities are cached.
    children_utilities = {}

    info_state_node = self._info_state_nodes[info_state]
    if policies is None:
      info_state_policy = self._get_infostate_policy(info_state)
    else:
      info_state_policy = policies[current_player](info_state)
    for action in state.legal_actions():
      action_prob = info_state_policy.get(action, 0.)
      new_state = state.child(action)
      new_reach_probabilities = reach_probabilities.copy()
      new_reach_probabilities[current_player] *= action_prob
      child_utility = self._compute_counterfactual_regret_for_player(
          new_state,
          policies=policies,
          reach_probabilities=new_reach_probabilities,
          player=player)

      state_value += action_prob * child_utility
      children_utilities[action] = child_utility

    # If we are performing alternating updates, and the current player is not
    # the current_player, we skip the cumulative values update.
    # If we are performing simultaneous updates, we do update the cumulative
    # values.
    simulatenous_updates = player is None
    if not simulatenous_updates and current_player != player:
      return state_value

    reach_prob = reach_probabilities[current_player]
    counterfactual_reach_prob = (
        np.prod(reach_probabilities[:current_player]) *
        np.prod(reach_probabilities[current_player + 1:]))
    state_value_for_player = state_value[current_player]

    for action, action_prob in info_state_policy.items():
      cfr_regret = counterfactual_reach_prob * (
          children_utilities[action][current_player] - state_value_for_player)

      info_state_node.cumulative_regret[action] += cfr_regret
      if self._linear_averaging:
        info_state_node.cumulative_policy[
            action] += self._iteration * reach_prob * action_prob
      else:
        info_state_node.cumulative_policy[action] += reach_prob * action_prob

    return state_value

  def _get_infostate_policy(self, info_state_str):
    """Returns an {action: prob} dictionary for the policy on `info_state`."""
    info_state_node = self._info_state_nodes[info_state_str]
    prob_vec = self._current_policy.action_probability_array[
        info_state_node.index_in_tabular_policy]
    return {
        action: prob_vec[action] for action in info_state_node.legal_actions
    }


def _regret_matching(cumulative_regrets, legal_actions):
  regrets = cumulative_regrets.values()
  sum_positive_regrets = sum((regret for regret in regrets if regret > 0))

  info_state_policy = {}
  if sum_positive_regrets > 0:
    for action in legal_actions:
      positive_action_regret = max(0.0, cumulative_regrets[action])
      info_state_policy[action] = (
          positive_action_regret / sum_positive_regrets)
  else:
    for action in legal_actions:
      info_state_policy[action] = 1.0 / len(legal_actions)
  return info_state_policy


class _CFRSolver(_CFRSolverBase):
  def evaluate_and_update_policy(self):
    """Performs a single step of policy evaluation and policy improvement."""
    self._iteration += 1
    if self._alternating_updates:
      for player in range(self._game.num_players()):
        self._compute_counterfactual_regret_for_player(
            self._root_node,
            policies=None,
            reach_probabilities=np.ones(self._game.num_players() + 1),
            player=player)
        if self._regret_matching_plus:
          _apply_regret_matching_plus_reset(self._info_state_nodes)
        _update_current_policy(self._current_policy, self._info_state_nodes)
    else:
      self._compute_counterfactual_regret_for_player(
          self._root_node,
          policies=None,
          reach_probabilities=np.ones(self._game.num_players() + 1),
          player=None)
      if self._regret_matching_plus:
        _apply_regret_matching_plus_reset(self._info_state_nodes)
      _update_current_policy(self._current_policy, self._info_state_nodes)


class CFRPlusSolver(_CFRSolver):
  def __init__(self, game):
    super(CFRPlusSolver, self).__init__(
        game,
        regret_matching_plus=True,
        alternating_updates=True,
        linear_averaging=True)


class CFRSolver(_CFRSolver):
  def __init__(self, game):
    super(CFRSolver, self).__init__(
        game,
        regret_matching_plus=False,
        alternating_updates=True,
        linear_averaging=False)
