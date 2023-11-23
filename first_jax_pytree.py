#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import jax
import jax.numpy as jnp
from jax import random

def predict_pytree(params, x):
    return jnp.dot(x, params['W']) + params['b']

def mse_pytree(params, x_batched, y_batched):
    def squared_error(x, y):
        y_pred = predict_pytree(params, x)
        return jnp.inner(y-y_pred, y-y_pred)/2.0
    return jnp.mean(jax.vmap(squared_error)(x_batched, y_batched), axis=0)

n_samples = 20
x_dim = 10
y_dim = 5

key = random.PRNGKey(0)
k1, k2 = random.split(key)
W = random.normal(k1, (x_dim, y_dim))
b = random.normal(k2, (y_dim,))

params = {'W': jnp.zeros_like(W), 'b': jnp.zeros_like(b)}

key_sample, key_noise = random.split(k1)
x_samples = random.normal(key_sample, (n_samples, x_dim))
y_samples = predict_pytree(params, x_samples) + 0.1 * random.normal(key_noise,(n_samples, y_dim))
print('x shape: ', x_samples.shape, ' y shape: ', y_samples.shape)


@jax.jit
def update_params_pytree(params, learning_rate, x_samples, y_samples):
    params = jax.tree_util.tree_map(
        lambda p, g: p - learning_rate * g, params, 
        jax.grad(mse_pytree)(params, x_samples, y_samples))
    return params

learning_rate = 0.3
print('Loss for W and b: ', mse_pytree({'W': W, 'b': b}, x_samples, y_samples))

for i in range(101):
    params = update_params_pytree(params, learning_rate, x_samples, y_samples)
    if i % 5 == 0:
        print('Loss: ', mse_pytree(params, x_samples, y_samples))
