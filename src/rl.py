#!/usr/bin/env python3

from collections import defaultdict

__author__ = "Dibyo Majumdar"
__email__ = "dibyo.majumdar@gmail.com"


class MDP:
    def __init__(self, states: list, actions: list, T: dict, R: dict, gamma: float):
        self.states = states
        self.actions = actions
        self.T = T
        self.R = R
        self.gamma = gamma


def value_iteration(mdp: MDP, horizon: float) -> (dict, dict):
    values = defaultdict(float)
    pi = {}

    for _ in range(horizon):
        values_new = {}
        for state in mdp.states:
            values_new[state], pi[state] = value_update(mdp, values, state)
        values = values_new

    return values, pi


def value_update(mdp: MDP, old_values: dict, state: object) -> (float, object):
    value_for_action = {}
    for action in mdp.actions:
        value = 0
        for new_state in mdp.states:
            value += mdp.T[(state, action, new_state)] * \
                     (mdp.R[(state, action, new_state)] + mdp.gamma*old_values[new_state])
        value_for_action[action] = value
    
    best_action, max_value = max(iter(value_for_action.items()), key=lambda item: item[1])
    return max_value, best_action