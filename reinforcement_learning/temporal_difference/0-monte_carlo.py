#!/usr/bin/env python3
"""Function monte carlo."""
import numpy as np
import gym


def monte_carlo(env, V, policy, episodes=5000, max_steps=100,
                alpha=0.1, gamma=0.99):
    """[summary]

    Args:
        env ([type]): [description]
        V ([type]): [description]
        policy ([type]): [description]
        episodes (int, optional): [description]. Defaults to 5000.
        max_steps (int, optional): [description]. Defaults to 100.
        alpha (float, optional): [description]. Defaults to 0.1.
        gamma (float, optional): [description]. Defaults to 0.99.

    Returns:
        [type]: [description]
    """
    env.reset()
    discount_factor = gamma ** np.arange(max_steps)
    for _ in range(episodes):
        states = [] 
        rewards = []
        init_state = env.reset()
        states.append(init_state)
        for _ in range(max_steps):
            action = policy(states[-1])
            state, reward, finished, _ = env.step(action)
            states.append(state)
            rewards.append(reward)
            if finished:
                break
        for idx in range(len(states)):
            state = states[idx]
            if idx == len(rewards):
                break
            rews_from_state = np.array(rewards)
            num_rews = rews_from_state.shape[0]
            disc_facts = discount_factor[:num_rews]
            disc_reward = (
                rews_from_state * disc_facts
                ).sum()
            V[state] = V[state] + alpha * (disc_reward - V[state])
    return V
