import pandas as pd 
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt 
from sklearn.metrics import r2_score


df_train = pd.read_csv('/home/alisson/Dev/machine_learning/linear_regression/data/train.csv')
df_test = pd.read_csv('/home/alisson/Dev/machine_learning/linear_regression/data/test.csv')

x_train = df_train['x']
y_train = df_train['y']
x_test = df_test['x']
y_test = df_test['y']

x_train = jnp.asarray(x_train)
y_train = jnp.asarray(y_train)
x_test = jnp.asarray(x_test)
y_test = jnp.asarray(y_test)


params = { 
    'w' : jnp.zeros(x_train.shape[1:]),
    'b' : 0.
    }
    
p

