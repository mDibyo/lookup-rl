#!/usr/bin/env python3

from collections import defaultdict
from enum import Enum
import json

__author__ = "Dibyo Majumdar"
__email__ = "dibyo.majumdar@gmail.com"


INF = float('inf')


class GridActions(Enum):
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
    def from_grid_file(cls, filename, goal_states, noise=0, gamma=1):
        with open(filename) as grid_file:
            return cls.from_grid(json.load(grid_file), goal_states, noise, gamma)

    @classmethod
    def from_grid(cls, rewards_grid, goal_states, noise=0, gamma=1):
        l = len(rewards_grid)
        b = len(rewards_grid[0])

        # Initialize MDP
        actions = [GridActions.up, GridActions.down, GridActions.left, GridActions.right]
        states = set()
        T = defaultdict(float)
        R = defaultdict(float)

        sink_state = 'hakuna matata'
        states.add(sink_state)
        max_reward = - INF

        # Set up rewards and transitions for normal states
        blocked_states = []
        for i, row in enumerate(rewards_grid):
            for j, reward in enumerate(row):
                states.add((i, j))
                if reward == 'X':  # a wall
                    blocked_states.append((i, j))
                    continue
                reward = float(reward)
                max_reward = max(max_reward, reward)
                if i > 0:  # top
                    T[((i-1, j), GridActions.down, (i, j))] = 1 - noise
                    T[((i-1, j), GridActions.up, (i, j))] = noise/3
                    T[((i-1, j), GridActions.left, (i, j))] = noise/3
                    T[((i-1, j), GridActions.right, (i, j))] = noise/3
                    R[((i-1, j), GridActions.down, (i, j))] = reward
                    R[((i-1, j), GridActions.up, (i, j))] = reward
                    R[((i-1, j), GridActions.left, (i, j))] = reward
                    R[((i-1, j), GridActions.right, (i, j))] = reward
                if i < l:  # bottom
                    T[((i+1, j), GridActions.up, (i, j))] = 1 - noise
                    T[((i+1, j), GridActions.down, (i, j))] = noise/3
                    T[((i+1, j), GridActions.left, (i, j))] = noise/3
                    T[((i+1, j), GridActions.right, (i, j))] = noise/3
                    R[((i+1, j), GridActions.up, (i, j))] = reward
                    R[((i+1, j), GridActions.down, (i, j))] = reward
                    R[((i+1, j), GridActions.left, (i, j))] = reward
                    R[((i+1, j), GridActions.right, (i, j))] = reward
                if j > 0:  # left
                    T[((i, j-1), GridActions.right, (i, j))] = 1 - noise
                    T[((i, j-1), GridActions.up, (i, j))] = noise/3
                    T[((i, j-1), GridActions.down, (i, j))] = noise/3
                    T[((i, j-1), GridActions.left, (i, j))] = noise/3
                    R[((i, j-1), GridActions.right, (i, j))] = reward
                    R[((i, j-1), GridActions.up, (i, j))] = reward
                    R[((i, j-1), GridActions.down, (i, j))] = reward
                    R[((i, j-1), GridActions.left, (i, j))] = reward
                if j < b:  # right
                    T[((i, j+1), GridActions.left, (i, j))] = 1 - noise
                    T[((i, j+1), GridActions.up, (i, j))] = noise/3
                    T[((i, j+1), GridActions.down, (i, j))] = noise/3
                    T[((i, j+1), GridActions.right, (i, j))] = noise/3
                    R[((i, j+1), GridActions.left, (i, j))] = reward
                    R[((i, j+1), GridActions.up, (i, j))] = reward
                    R[((i, j+1), GridActions.down, (i, j))] = reward
                    R[((i, j+1), GridActions.right, (i, j))] = reward

        # Set up rewards and transitions for goal states and blocked states
        for i, j in (goal_states + blocked_states):
            T[((i, j), GridActions.up, sink_state)] = 1
            T[((i, j), GridActions.down, sink_state)] = 1
            T[((i, j), GridActions.left, sink_state)] = 1
            T[((i, j), GridActions.right, sink_state)] = 1
            R[((i, j), GridActions.up, sink_state)] = max_reward * 2
            R[((i, j), GridActions.down, sink_state)] = max_reward * 2
            R[((i, j), GridActions.left, sink_state)] = max_reward * 2
            R[((i, j), GridActions.right, sink_state)] = max_reward * 2
            if i > 0:  # top
                T[((i, j), GridActions.up, (i-1, j))] = 0
                T[((i, j), GridActions.down, (i-1, j))] = 0
                T[((i, j), GridActions.left, (i-1, j))] = 0
                T[((i, j), GridActions.right, (i-1, j))] = 0
                R[((i, j), GridActions.up, (i-1, j))] = 0
                R[((i, j), GridActions.down, (i-1, j))] = 0
                R[((i, j), GridActions.left, (i-1, j))] = 0
                R[((i, j), GridActions.right, (i-1, j))] = 0
            if i < l:  # bottom
                T[((i, j), GridActions.up, (i+1, j))] = 0
                T[((i, j), GridActions.down, (i+1, j))] = 0
                T[((i, j), GridActions.left, (i+1, j))] = 0
                T[((i, j), GridActions.right, (i+1, j))] = 0
                R[((i, j), GridActions.up, (i+1, j))] = 0
                R[((i, j), GridActions.down, (i+1, j))] = 0
                R[((i, j), GridActions.left, (i+1, j))] = 0
                R[((i, j), GridActions.right, (i+1, j))] = 0
            if j > 0:  # left
                T[((i, j), GridActions.up, (i, j-1))] = 0
                T[((i, j), GridActions.down, (i, j-1))] = 0
                T[((i, j), GridActions.left, (i, j-1))] = 0
                T[((i, j), GridActions.right, (i, j-1))] = 0
                R[((i, j), GridActions.up, (i, j-1))] = 0
                R[((i, j), GridActions.down, (i, j-1))] = 0
                R[((i, j), GridActions.left, (i, j-1))] = 0
                R[((i, j), GridActions.right, (i, j-1))] = 0
            if j < b:  # right
                T[((i, j), GridActions.up, (i, j+1))] = 0
                T[((i, j), GridActions.down, (i, j+1))] = 0
                T[((i, j), GridActions.left, (i, j+1))] = 0
                T[((i, j), GridActions.right, (i, j+1))] = 0
                R[((i, j), GridActions.up, (i, j+1))] = 0
                R[((i, j), GridActions.down, (i, j+1))] = 0
                R[((i, j), GridActions.left, (i, j+1))] = 0
                R[((i, j), GridActions.right, (i, j+1))] = 0

        return cls(states, actions, T, R, gamma)


def generate_rewards_grid(grid, rewards_func=lambda i, j: 1):
    rewards_grid = grid[:]
    for i in range(len(grid)):
        for j in range(len(grid[0])):
            if grid[i][j] == 'X':
                rewards_grid[i][j] = 'X'
            else:
                rewards_grid[i][j] = rewards_func(i, j)
    return grid


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
