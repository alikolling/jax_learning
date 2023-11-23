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
import flax
import optax
import numpy as np
import jax.numpy as jnp
from flax import linen as nn
from flax.training import checkpoints
from flax.training import train_state
import torch.utils.data as data

class XORDataset(data.Dataset):
    
    def __init__(self, size, seed, std=0.1):
        '''
        Inputs:
            size - Number of data points we want to generate.
            seed - The seed to use to create thr PRNG state with wich we want to generate the data points.
            std -  Standart deviation of the noise (see generate_continuous_xor function)
        '''
        super().__init__()
        self.size = size
        self.np_rng = np.random.RandomState(seed=seed)
        self.std = std
        self.generate_continuous_xor()

    def generate_continuous_xor(self):
        # Each data point in the XOR dataset has two variables, x and y, that can be either 0 or 1
        # The label is their XOR combination, i.e. 1 if only x or only y is 1 while the other is 0.
        # If x=y, the label is 0.
        data = self.np_rng.randint(low=0, high=2, size=(self.size, 2)).astype(np.float32)
        label = (data.sum(axis=1) == 1).astype(np.float32)
        # To make it slightly more challenging, we add a bit of gaussian noise to the data points.
        data += self.np_rng.normal(loc=0.0, scale=self.std, size=data.shape)

        self.data = data
        self.label = label

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        # Return the idx-th data point of the dataset
        # If we have multiple things to return (data point and label), we can return them as tuple
        data_point = self.data[idx]
        data_label = self.label[idx]

        return data_point, data_label


class SimpleClassifier(nn.Module):
    num_hiddens : int
    num_outputs : int

    def setup(self):

        self.linear1 = nn.Dense(features=self.num_hiddens)
        self.linear2 = nn.Dense(features=self.num_outputs)

    def __call__(self, x):

        x = self.linear1(x)
        x = nn.tanh(x)
        x = self.linear2(x)
        return x

class SimpleClassifierCompact(nn.Module):
    num_hidden : int   # Number of hidden neurons
    num_outputs : int  # Number of output neurons

    @nn.compact  # Tells Flax to look for defined submodules
    def __call__(self, x):
        # Perform the calculation of the model to determine the prediction
        # while defining necessary layers
        x = nn.Dense(features=self.num_hidden)(x)
        x = nn.tanh(x)
        x = nn.Dense(features=self.num_outputs)(x)
        return x

model = SimpleClassifier(num_hiddens=8, num_outputs=1)
# Printing the model shows its attributes
print(model)

rng = jax.random.PRNGKey(42)

rng, inp_rng, init_rng = jax.random.split(rng, 3)
inp = jax.random.normal(inp_rng, (8,2))

params = model.init(init_rng, inp)
print(params)

dataset = XORDataset(size=200, seed=42)
print('Size of dataset:', len(dataset))
print('data: ', dataset[0])

def visualize_samples(data, label):
    data_0 = data[label == 0]
    data_1 = data[label == 1]

    plt.figure(figsize=(4,4))
    plt.scatter(data_0[:,0],data_0[:,1], edgecolors="#333", label="Class 0")
    plt.scatter(data_1[:,0],data_1[:,1], edgecolors="#333", label="Class 1")
    plt.title("Dataset samples")
    plt.ylabel(r"$x_2$")
    plt.xlabel(r"$x_1$")
    plt.legend()

visualize_samples(dataset.data, dataset.label)
plt.show()

# This collate function is taken from the JAX tutorial with PyTorch Data Loading
# https://jax.readthedocs.io/en/latest/notebooks/Neural_Network_and_Data_Loading.html
def numpy_collate(batch):
    if isinstance(batch[0], np.ndarray):
        return np.stack(batch)
    if isinstance(batch[0], (tuple,list)):
        transposed = zip(*batch)
        return [numpy_collate(samples) for samples in transposed]
    else:
        return  np.array(batch)

data_loader = data.DataLoader(dataset, batch_size=8, shuffle=True, collate_fn=numpy_collate)

optimizer = optax.sgd(learning_rate=0.1)

model_state = train_state.TrainState.create(apply_fn=model.apply,
                                            params=params,
                                            tx=optimizer)

def calculate_loss_acc(state, params, batch):
    data_input, labels = batch
    # Obtain the logits and predictions of the model for the input data
    logits = state.apply_fn(params, data_input).squeeze(axis=-1)
    pred_labels = (logits > 0).astype(jnp.float32)
    # Calculate the loss and accuracy
    loss = optax.sigmoid_binary_cross_entropy(logits, labels).mean()
    acc = (pred_labels == labels).mean()

    return loss, acc

@jax.jit
def train_step(state, batch):
    # Gradient function
    grad_fn = jax.value_and_grad(calculate_loss_acc, #Function to calculate the loss
                                 argnums=1, # parameters are the second argument of the function
                                 has_aux=True # Function has additional outputs, here accuracy
                                 )
    # Determine gradients for current model, parameters and batch
    (loss, acc), grads = grad_fn(state, state.params, batch)
    # Perform parameter update with gradients and optimizer
    state = state.apply_gradients(grads=grads)
    #return state and any other value we might want
    return state, loss, acc

@jax.jit
def eval_step(state, batch):
    #determine accuracy
    _, acc = calculate_loss_acc(state, state.params, batch)
    return acc

train_dataset = XORDataset(size=2500, seed=42)
train_data_loader = data.DataLoader(train_dataset, batch_size=128, shuffle=True, collate_fn=numpy_collate)

def train_model(state, data_loader, num_epochs=100):
    #training loop
    for epoch in tqdm(range(num_epochs)):
        for batch in data_loader:
            state, loss, acc = train_step(state, batch)
           
    return state

trained_model_state = train_model(model_state, data_loader, 100)

checkpoints.save_checkpoint(ckpt_dir='my_checkpoints',
                            target=trained_model_state,
                            step=100,
                            prefix='my_model',
                            overwrite=True)

loaded_model_state = checkpoints.restore_checkpoint(ckpt_dir='my_checkpoints',
                                                    target=model_state,
                                                    prefix='my_model')

test_dataset = XORDataset(size=500, seed=123)
test_data_loader = data.DataLoader(test_dataset,
                                   batch_size=128,
                                   shuffle=False,
                                   drop_last=False,
                                   collate_fn=numpy_collate)

def eval_model(state, data_loader):
    all_accs, batch_sizes = [], []
    for batch in data_loader:
        batch_acc = eval_step(state, batch)
        all_accs.append(batch_acc)
        batch_sizes.append(batch[0].shape[0])
    acc = sum([a*b for a,b in zip(all_accs, batch_sizes)])/ sum(batch_sizes)
    print(f"Accuracy of the model: {100.0*acc:4.2f}%")

eval_model(trained_model_state, test_data_loader)

trained_model = model.bind(trained_model_state.params)

data_input, labels = next(iter(test_data_loader))
out = trained_model(data_input)
print(out)

def visualize_classification(model, data, label):
    data_0 = data[label == 0]
    data_1 = data[label == 1]

    fig = plt.figure(figsize=(4,4), dpi=500)
    plt.scatter(data_0[:,0], data_0[:,1], edgecolor="#333", label='Class 0')
    plt.scatter(data_1[:,0], data_1[:,1], edgecolor="#333", label='Class 1')
    plt.title("Dataset samples")
    plt.ylabel(r"$x_2$")
    plt.xlabel(r"$x_1$")
    plt.legend()

    c0 = np.array(to_rgba('C0'))
    c1 = np.array(to_rgba('C1'))
    x1 = jnp.arange(-0.5, 1.5, step=0.01)
    x2 = jnp.arange(-0.5, 1.5, step=0.01)
    xx1, xx2 = jnp.meshgrid(x1, x2, indexing='ij')
    model_inputs = np.stack([xx1, xx2], axis=-1)
    logits = model(model_inputs)
    preds = nn.sigmoid(logits)
    output_image = (1 - preds) * c0[None, None] + preds * c1[None, None] # specifying "None" in a dimension creates a new one
    output_image = jax.device_get(output_image)
    plt.imshow(output_image, origin='lower', extent=(-0.5, 1.5, -0.5, 1.5))
    plt.grid(False)
    return fig

_ = visualize_classification(trained_model, dataset.data, dataset.label)
plt.show()

