#!/usr/bin/env python
# -*- coding: utf-8 -*-

import jax
import numpy as onp
import jax.numpy as jnp
from jax import grad, jit, vmap, value_and_grad
from jax import random

def predict(W, b, x):
    return jnp.dot(x, W) + b

def mse(W, b, x_batch, y_batch):

    def squared_error(x, y):
        y_pred = predict(W, b, x)
        return jnp.inner(y-y_pred,y-y_pred)/2.
    return jnp.mean(jax.vmap(squared_error)(x_batch, y_batch), axis=0)

n_samples = 20
x_dim = 10
y_dim = 5

key = random.PRNGKey(0)
k1, k2 = random.split(key)
W = random.normal(k1, (x_dim, y_dim))
b = random.normal(k2, (y_dim,))

key_sample, key_noise = random.split(k1)
x_samples = random.normal(key_sample, (n_samples, x_dim))
y_samples = predict(W, b, x_samples) + 0.1 * random.normal(key_noise,(n_samples, y_dim))
print('x shape: ', x_samples.shape, ' y shape: ', y_samples.shape)


W_hat = jnp.zeros_like(W)
b_hat = jnp.zeros_like(b)

@jax.jit
def update_params(W, b, x, y, lr):
    W, b = W - lr * jax.grad(mse, 0)(W, b, x, y), b - lr * jax.grad(mse, 1)(W, b, x, y)
    return W, b

learning_rate = 0.3
print('loss for W and b: ', mse(W, b, x_samples, y_samples))

for i in range(101):
    W_hat, b_hat = update_params(W_hat, b_hat, x_samples, y_samples, learning_rate)

    if i % 5 == 0:
        print('Loss: ', mse(W_hat, b_hat, x_samples, y_samples))



