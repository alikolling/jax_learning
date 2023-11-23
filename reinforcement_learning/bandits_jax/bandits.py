#!/usr/bin/env python
# -*- coding: utf-8 -*-

import jax
import jax.numpy as jnp
from jax import random
from jax.tree_util import tree_map

import flax
from flax import linen as nn
from flax.training import train_state, checkpoints


class ContextBandit:

    def __init__(self, arms=10):
        self.arms = arms
        self.key = random.PRNGKey(0)
        self.init_distributions(arms)
        self.update_state()

    def init_distributions(self, arms):
        self.bandit_matrix = random.normal(self.key, (arms, arms))

    def reward(self, prob):
        reward = 0
        for i in range(self.arms):
            if [random.normal(self.key, shape=(1, )) < prob]:
                reward += 1
        return reward

    def get_state(self):
        return self.state

    def update_state(self):
        self.state = random.randint(self.key,
                                    shape=(1, ),
                                    minval=0,
                                    maxval=self.arms)

    def get_reward(self, arm):
        return self.reward(self.bandit_matrix[self.get_state()][arm])

    def choose_arm(self, arm):
        reward = self.get_reward(arm)
        self.update_state()
        return reward
