#!/usr/bin/env python3
"""[summary]

Returns:
    [type]: [description]
"""
import numpy as np
import gym


def greedy_policy_action(epsilon, state, Q):
    """[summary]

    Args:
        epsilon ([type]): [description]
        state ([type]): [description]
        Q ([type]): [description]

    Returns:
        [type]: [description]
    """
    num_actions = Q.shape[1]
    r = np.random.uniform(0.1)
    if r >= epsilon:
        return Q[state].argmax()
    else:
        return np.random.randint(0, num_actions)


def sarsa_lambtha(env, Q, lambtha,
                  episodes=5000, max_steps=100,
                  alpha=0.1,
                  gamma=0.99, epsilon=1,
                  min_epsilon=0.1, epsilon_decay=0.05):
    """[summary]

    Args:
        env ([type]): [description]
        Q ([type]): [description]
        lambtha ([type]): [description]
        episodes (int, optional): [description]. Defaults to 5000.
        max_steps (int, optional): [description]. Defaults to 100.
        alpha (float, optional): [description]. Defaults to 0.1.
        gamma (float, optional): [description]. Defaults to 0.99.
        epsilon (int, optional): [description]. Defaults to 1.
        min_epsilon (float, optional): [description]. Defaults to 0.1.
        epsilon_decay (float, optional): [description]. Defaults to 0.05.

    Returns:
        [type]: [description]
    """
    epsilon_0 = epsilon
    ET = np.zeros(Q.shape)
    for episode in range(episodes):
        last_state = env.reset()
        last_action = greedy_policy_action(epsilon, last_state, Q)
        for _ in range(max_steps):
            state, reward, finished, _ = env.step(last_action)
            action = greedy_policy_action(epsilon, last_state, Q)
            ET *= gamma * lambtha
            ET[last_state, last_action] = 1
            delta = (reward +
                     gamma * Q[state, action] - Q[last_state, last_action])
            Q[last_state, last_action] = (Q[last_state, last_action] +
                                          alpha * delta *
                                          ET[last_state, last_action])
            last_state = state
            last_action = action
            if finished:
                break
        epsilon = (min_epsilon + (epsilon_0 - min_epsilon) *
                   np.exp(-epsilon_decay * episode))
    return Q
