#!/usr/bin/env python3

from collections import defaultdict
from enum import Enum
import json

from operator import add
from functools import reduce


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


action_symbol_map = {
    GridActions.up: '↑',
    GridActions.down: '↓',
    GridActions.left: '←',
    GridActions.right: '→',
}


class MDP:
    def __init__(self, states: list, actions: list, T: dict, R: dict, gamma: float):
        self.states = states
        self.actions = actions
        self.T = T
        self.R = R
        self.gamma = gamma

    @classmethod
    def from_2d_grid_file(cls, filename, reward_func, goal_state, noise=0, gamma=1):
        with open(filename) as grid_file:
            return cls.from_2d_grid(json.load(grid_file), reward_func, goal_state, noise, gamma)

    @classmethod
    def from_2d_grid(cls, grid, reward_func, goal_state, noise=0, gamma=1):
        l = len(grid)
        b = len(grid[0])

        # Initialize MDP
        actions = [GridActions.up, GridActions.down, GridActions.left, GridActions.right]
        states = set()
        T = defaultdict(float)
        R = defaultdict(float)

        sink_state = 'hakuna matata'
        states.add(sink_state)

        # Set up rewards and transitions for normal states
        blocked_states = []
        for i, row in enumerate(grid):
            for j, token in enumerate(row):
                states.add((i, j))
                if token == 'X':  # a wall
                    blocked_states.append((i, j))
                    continue
                if i > 0:  # top
                    T[((i-1, j), GridActions.down, (i, j))] = 1 - noise
                    T[((i-1, j), GridActions.left, (i, j))] = noise/2
                    T[((i-1, j), GridActions.right, (i, j))] = noise/2
                    R[((i-1, j), GridActions.down, (i, j))] = reward_func((i-1, j), (i, j), goal_state)
                    R[((i-1, j), GridActions.up, (i, j))] = reward_func((i-1, j), (i, j), goal_state)
                    R[((i-1, j), GridActions.left, (i, j))] = reward_func((i-1, j), (i, j), goal_state)
                    R[((i-1, j), GridActions.right, (i, j))] = reward_func((i-1, j), (i, j), goal_state)
                if i < l:  # bottom
                    T[((i+1, j), GridActions.up, (i, j))] = 1 - noise
                    T[((i+1, j), GridActions.left, (i, j))] = noise/2
                    T[((i+1, j), GridActions.right, (i, j))] = noise/2
                    R[((i+1, j), GridActions.up, (i, j))] = reward_func((i+1, j), (i, j), goal_state)
                    R[((i+1, j), GridActions.down, (i, j))] = reward_func((i+1, j), (i, j), goal_state)
                    R[((i+1, j), GridActions.left, (i, j))] = reward_func((i+1, j), (i, j), goal_state)
                    R[((i+1, j), GridActions.right, (i, j))] = reward_func((i+1, j), (i, j), goal_state)
                if j > 0:  # left
                    T[((i, j-1), GridActions.right, (i, j))] = 1 - noise
                    T[((i, j-1), GridActions.up, (i, j))] = noise/2
                    T[((i, j-1), GridActions.down, (i, j))] = noise/2
                    R[((i, j-1), GridActions.right, (i, j))] = reward_func((i, j-1), (i, j), goal_state)
                    R[((i, j-1), GridActions.up, (i, j))] = reward_func((i, j-1), (i, j), goal_state)
                    R[((i, j-1), GridActions.down, (i, j))] = reward_func((i, j-1), (i, j), goal_state)
                    R[((i, j-1), GridActions.left, (i, j))] = reward_func((i, j-1), (i, j), goal_state)
                if j < b:  # right
                    T[((i, j+1), GridActions.left, (i, j))] = 1 - noise
                    T[((i, j+1), GridActions.up, (i, j))] = noise/2
                    T[((i, j+1), GridActions.down, (i, j))] = noise/2
                    R[((i, j+1), GridActions.left, (i, j))] = reward_func((i, j+1), (i, j), goal_state)
                    R[((i, j+1), GridActions.up, (i, j))] = reward_func((i, j+1), (i, j), goal_state)
                    R[((i, j+1), GridActions.down, (i, j))] = reward_func((i, j+1), (i, j), goal_state)
                    R[((i, j+1), GridActions.right, (i, j))] = reward_func((i, j+1), (i, j), goal_state)

        # Set up rewards and transitions for goal states and blocked states
        sink_reward = 0  # l * b
        for i, j in ([goal_state] + blocked_states):
            T[((i, j), GridActions.up, sink_state)] = 1
            T[((i, j), GridActions.down, sink_state)] = 1
            T[((i, j), GridActions.left, sink_state)] = 1
            T[((i, j), GridActions.right, sink_state)] = 1
            R[((i, j), GridActions.up, sink_state)] = sink_reward
            R[((i, j), GridActions.down, sink_state)] = sink_reward
            R[((i, j), GridActions.left, sink_state)] = sink_reward
            R[((i, j), GridActions.right, sink_state)] = sink_reward
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

    return dict(values), pi


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


def lookup_policy(mdp, F_m, F_n):
    """
    :param F_m: (m, V_m, C_m)
    :param F_n: (n, V_n, C_n)
    :return: pi_lookup { mdp.state -> m|n}
    """
    m, V_m, C_m = F_m
    n, V_n, C_n = F_n

    pi_lookup = {}
    for state in mdp.states:
        u_m = V_m[state] - C_m
        u_n = V_n[state] - C_n
        pi_lookup[state] = m if u_m >= u_n else n

    return pi_lookup


def l1_norm(p1, p2):
    return reduce(add, map(lambda x: abs(x[0]-x[1]), zip(p1, p2)), 0)


def l1_norm_reward(old_state, new_state, goal_state):
    return l1_norm(old_state, goal_state) - l1_norm(new_state, goal_state)


def lookup_policy_proportional(mdp, C, m, n):
    V_m, _ = value_iteration(mdp, m)
    V_n, _ = value_iteration(mdp, n)
    return lookup_policy(mdp, (m, V_m, C*m), (n, V_n, C*n))


def visualize_2d_grid_policy(pi_lookup, grid, symbol_map):
    policy_grid = grid[:]
    for i, row in enumerate(grid):
        for j, token in enumerate(row):
            if token != 'X':
                policy_grid[i][j] = symbol_map[pi_lookup[(i, j)]]
    return policy_grid


class FunctionDict:
    def __init__(self, func):
        self.func = func

    def __getitem__(self, item):
        return self.func(item)


if __name__ == '__main__':
    grid = json.load(open('../data/grids/long_template.json'))
    mdp = MDP.from_2d_grid_file('../data/grids/long_template.json', l1_norm_reward, (21, 3), )
    pi_lookup = lookup_policy_proportional(mdp, 0.8, 1, 10)
    visualize_2d_grid_policy(pi_lookup, grid, {1: '_', 10: '*'})

    # Debugging
    # id_func = lambda x: x
    # v1, pi1 = value_iteration(mdp, 1)
    # v10, pi10 = value_iteration(mdp, 10)
    # visualize_2d_grid_policy(pi10, grid, action_symbol_map)
    # visualize_2d_grid_policy(v10, grid, id_func)
