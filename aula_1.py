#!/usr/bin/env python
# -*- coding: utf-8 -*-


## Standard libraries
import os
import math
import numpy as np
import time

## Imports for plotting
import matplotlib.pyplot as plt

from IPython.display import set_matplotlib_formats
set_matplotlib_formats('svg', 'pdf') # For export
from matplotlib.colors import to_rgba
import seaborn as sns
sns.set()

## Progress bar
from tqdm.auto import tqdm

import jax
import jax.numpy as jnp

a = jnp.zeros((2,5), dtype=jnp.float32)
print(a)

b = jnp.arange(6)
print(b)

print(b.__class__)
print(b.device())
b_cpu = jax.device_get(b)
print(b_cpu.__class__)

b_new = b.at[0].set(1)
print('Original array:', b)
print('Changed array:', b_new)

def simple_graph(x):
    x = x + 2
    x = x ** 2
    x = x + 3
    y = x.mean()
    return y

inp = jnp.arange(3, dtype=jnp.float32)
print('Input', inp)
print('Output', simple_graph(inp))

grad_function = jax.grad(simple_graph)
gradients = grad_function(inp)
print('Gradient', gradients)



