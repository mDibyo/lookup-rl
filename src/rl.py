#!/usr/bin/env python3

from collections import defaultdict
from enum import Enum

__author__ = "Dibyo Majumdar"
__email__ = "dibyo.majumdar@gmail.com"


class Actions(Enum):
    up = 1
    down = 2
    left = 3
    right = 4
    top = 5
    bottom = 6


class MDP:
    def __init__(self, states: list, actions: list, T: dict, R: dict, gamma: float):
        self.states = states
        self.actions = actions
        self.T = T
        self.R = R
        self.gamma = gamma

    @classmethod
    def from_grid(cls, rewards_grid, goal_states, noise=0, gamma=1):
        l = len(rewards_grid)
        b = len(rewards_grid[0])

        # Initialize MDP
        actions = [Actions.up, Actions.down, Actions.left, Actions.right]
        states = set()
        T = defaultdict(float)
        R = defaultdict(float)

        # Set up rewards and transitions
        for i, row in enumerate(rewards_grid):
            for j, reward in enumerate(row):
                states.add((i, j))
                if reward != 'X':
                    reward = float(reward)
                    if i > 0:  # top
                        T[((i-1, j), Actions.down, (i, j))] = 1 - noise
                        T[((i-1, j), Actions.up, (i, j))] = noise/3
                        T[((i-1, j), Actions.left, (i, j))] = noise/3
                        T[((i-1, j), Actions.right, (i, j))] = noise/3
                        R[((i-1, j), Actions.down, (i, j))] = reward
                        R[((i-1, j), Actions.up, (i, j))] = reward
                        R[((i-1, j), Actions.left, (i, j))] = reward
                        R[((i-1, j), Actions.right, (i, j))] = reward
                    if i < l:  # bottom
                        T[((i+1, j), Actions.up, (i, j))] = 1 - noise
                        T[((i+1, j), Actions.down, (i, j))] = noise/3
                        T[((i+1, j), Actions.left, (i, j))] = noise/3
                        T[((i+1, j), Actions.right, (i, j))] = noise/3
                        R[((i+1, j), Actions.up, (i, j))] = reward
                        R[((i+1, j), Actions.down, (i, j))] = reward
                        R[((i+1, j), Actions.left, (i, j))] = reward
                        R[((i+1, j), Actions.right, (i, j))] = reward
                    if j > 0:  # left
                        T[((i, j-1), Actions.right, (i, j))] = 1 - noise
                        T[((i, j-1), Actions.up, (i, j))] = noise/3
                        T[((i, j-1), Actions.down, (i, j))] = noise/3
                        T[((i, j-1), Actions.left, (i, j))] = noise/3
                        R[((i, j-1), Actions.right, (i, j))] = reward
                        R[((i, j-1), Actions.up, (i, j))] = reward
                        R[((i, j-1), Actions.down, (i, j))] = reward
                        R[((i, j-1), Actions.left, (i, j))] = reward
                    if j < b:  # right
                        T[((i, j+1), Actions.left, (i, j))] = 1 - noise
                        T[((i, j+1), Actions.up, (i, j))] = noise/3
                        T[((i, j+1), Actions.down, (i, j))] = noise/3
                        T[((i, j+1), Actions.right, (i, j))] = noise/3
                        R[((i, j+1), Actions.left, (i, j))] = reward
                        R[((i, j+1), Actions.up, (i, j))] = reward
                        R[((i, j+1), Actions.down, (i, j))] = reward
                        R[((i, j+1), Actions.right, (i, j))] = reward

        for i, j in goal_states:
            T[((i, j), Actions.up, (i, j))] = 1
            T[((i, j), Actions.down, (i, j))] = 1
            T[((i, j), Actions.left, (i, j))] = 1
            T[((i, j), Actions.right, (i, j))] = 1
            R[((i, j), Actions.up, (i, j))] = 0
            R[((i, j), Actions.down, (i, j))] = 0
            R[((i, j), Actions.left, (i, j))] = 0
            R[((i, j), Actions.right, (i, j))] = 0
            if i > 0:  # top
                T[((i, j), Actions.up, (i-1, j))] = 0
                T[((i, j), Actions.down, (i-1, j))] = 0
                T[((i, j), Actions.left, (i-1, j))] = 0
                T[((i, j), Actions.right, (i-1, j))] = 0
                R[((i, j), Actions.up, (i-1, j))] = 0
                R[((i, j), Actions.down, (i-1, j))] = 0
                R[((i, j), Actions.left, (i-1, j))] = 0
                R[((i, j), Actions.right, (i-1, j))] = 0
            if i < l:  # bottom
                T[((i, j), Actions.up, (i+1, j))] = 0
                T[((i, j), Actions.down, (i+1, j))] = 0
                T[((i, j), Actions.left, (i+1, j))] = 0
                T[((i, j), Actions.right, (i+1, j))] = 0
                R[((i, j), Actions.up, (i+1, j))] = 0
                R[((i, j), Actions.down, (i+1, j))] = 0
                R[((i, j), Actions.left, (i+1, j))] = 0
                R[((i, j), Actions.right, (i+1, j))] = 0
            if j > 0:  # left
                T[((i, j), Actions.up, (i, j-1))] = 0
                T[((i, j), Actions.down, (i, j-1))] = 0
                T[((i, j), Actions.left, (i, j-1))] = 0
                T[((i, j), Actions.right, (i, j-1))] = 0
                R[((i, j), Actions.up, (i, j-1))] = 0
                R[((i, j), Actions.down, (i, j-1))] = 0
                R[((i, j), Actions.left, (i, j-1))] = 0
                R[((i, j), Actions.right, (i, j-1))] = 0
            if j < b:  # right
                T[((i, j), Actions.up, (i, j+1))] = 0
                T[((i, j), Actions.down, (i, j+1))] = 0
                T[((i, j), Actions.left, (i, j+1))] = 0
                T[((i, j), Actions.right, (i, j+1))] = 0
                R[((i, j), Actions.up, (i, j+1))] = 0
                R[((i, j), Actions.down, (i, j+1))] = 0
                R[((i, j), Actions.left, (i, j+1))] = 0
                R[((i, j), Actions.right, (i, j+1))] = 0
        return cls(states, actions, T, R, gamma)


def value_iteration(mdp: MDP, horizon: int=1) -> (dict, dict):
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
