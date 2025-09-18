import numpy as np
from open_spiel.python import policy as os_policy
from open_spiel.python.algorithms import exploitability


def regret_matching(regret_vec: np.ndarray):
    pos = np.maximum(regret_vec, 0.0)
    s = pos.sum()
    if s > 0:
        return pos / s
    else:
        return np.ones_like(pos) / len(pos)


def infoset_key(state, player):
    return state.information_state_string(player)


def legal_action_index_map(legal_actions):
    return {a: i for i, a in enumerate(legal_actions)}


def to_tabular_policy(game, avg_strategies_by_player):

    tab = os_policy.TabularPolicy(game)

    state = game.new_initial_state()
    stack = [state]
    visited = set()
    while stack:
        s = stack.pop()
        if s.is_terminal():
            continue
        if s.is_chance_node():
            for a, _ in s.chance_outcomes():
                stack.append(s.child(a))
            continue
        p = s.current_player()
        key = infoset_key(s, p)
        la = s.legal_actions()
        if key in avg_strategies_by_player[p]:
            probs = avg_strategies_by_player[p][key]
            idx_map = legal_action_index_map(la)
            row = tab.policy_for_key((s.information_state_string(p), p))
            for i in range(len(row)):
                row[i] = 0.0

            for i, a in enumerate(la):
                row[i] = float(probs[i])
        # DFS
        for a in la:
            child = s.child(a)

            h = child.history_str()
            if h not in visited:
                visited.add(h)
                stack.append(child)
    return tab
