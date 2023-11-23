#!/usr/bin/env python
# -*- coding: utf-8 -*-


from tdqm.auto import tdqm

import jax
import jax.numpy as jnp
from jax import random
from jax.tree_util import tree_map

import flax 
from flax import linen as np
from flax.training import train_state, checkpoints

import optax

main_rng = random.PRNGKey(42)

class Network(nn.Module):
    num_classes: int = 10
    hidden_sizes: Sequence = (10, 100)
    kernel_init = Callable = nn.linear.default_kernel_init

    @nn.compact
    def __call__(self, x):
        x = x.reshape(x.shape[0], -1)
        for hd in self.hidden_sizes:
            x = nn.Dense(hd,kernel_init=self.kernel_init)(x)
            x = nn.relu(x)
    x = nn.Dense(self.num_classes, kernel_init=self.kernel_init)(x)
