#!/usr/bin/env python
# -*- coding: utf-8 -*-

from tqdm.auto import tqdm
from typing import Sequence, Any
from clu import metrics
from flax import struct
import jax
import jax.numpy as jnp
from jax import random
from jax.tree_util import tree_map
import flax
from flax import linen as nn
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
            x = nn.Dense(hd, kernel_init=self.kernel_init)(x)
            x = nn.relu(x)

        x = nn.Dense(self.num_classes, kernel_init=self.kernel_init)(x)
        return x


@struct.dataclass
class Metrics(metrics.Collection):
    accuracy: metrics.Accuracy
    loss: metrics.Average.from_output('loss')


class TrainState(train_state.TrainState):
    metrics: Metrics


def create_train_state(module, rng, learning_rate, momentum):
    """Creates an initial `TrainState`."""
    params = module.init(rng, jnp.ones(
        [1, 28, 28,
         1]))['params']  # initialize parameters by passing a template image
    tx = optax.sgd(learning_rate, momentum)
    return TrainState.create(apply_fn=module.apply,
                             params=params,
                             tx=tx,
                             metrics=Metrics.empty())


@jax.jit
def train_step(state, batch):
    """Train for a single step."""

    def loss_fn(params):
        logits = state.apply_fn({'params': params}, batch['image'])
        loss = optax.softmax_cross_entropy_with_integer_labels(
            logits=logits, labels=batch['label']).mean()
        return loss

    grad_fn = jax.grad(loss_fn)
    grads = grad_fn(state.params)
    state = state.apply_gradients(grads=grads)
    return state


@jax.jit
def compute_metrics(*, state, batch):
    logits = state.apply_fn({'params': state.params}, batch['image'])
    loss = optax.softmax_cross_entropy_with_integer_labels(
        logits=logits, labels=batch['label']).mean()
    metric_updates = state.metrics.single_from_model_output(
        logits=logits, labels=batch['label'], loss=loss)
    metrics = state.metrics.merge(metric_updates)
    state = state.replace(metrics=metrics)
    return state
