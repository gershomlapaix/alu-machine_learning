#!/usr/bin/env python3
"""Function td_lambtha."""
import numpy as np
import gym


def td_lambtha(env, V, policy, lambtha,
               episodes=5000, max_steps=100,
               alpha=0.1, gamma=0.99):
    """[summary]

    Args:
        env ([type]): [description]
        V ([type]): [description]
        policy ([type]): [description]
        lambtha ([type]): [description]
        episodes (int, optional): [description]. Defaults to 5000.
        max_steps (int, optional): [description]. Defaults to 100.
        alpha (float, optional): [description]. Defaults to 0.1.
        gamma (float, optional): [description]. Defaults to 0.99.

    Returns:
        [type]: [description]
    """
    ET = np.zeros(V.shape)
    for _ in range(episodes):
        last_state = env.reset()
        for _ in range(max_steps):
            action = policy(last_state)
            state, reward, finished, _ = env.step(action)
            ET *= gamma * lambtha
            ET[last_state] = 1
            delta = reward + gamma * V[state] - V[last_state]
            V[last_state] = V[last_state] + alpha * delta * ET[last_state]
            last_state = state
            if finished:
                break
    return V
