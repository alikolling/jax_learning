import jax.numpy as jnp
from diffrax import ODETerm, Tsit5

vector_field = lambda t, x, args: -y
term = ODETerm(vector_field)
solver = Tsit5()

t0 = 0
dt0 = 0.05
t1 = 1
y0 = jnp.array(1.0)
args = None

tprev = t0
tnext = t0 + dt0
y = y0
state = solver.init(term, tprev, tnext, y0, args)

while tprev < t1:
    y, _, _, state, _ = solver.step(term, tprev, tnext, y, args, state, made_jump=False)
    print(f"T = {tnext} Y = {y}")
    tprev = tnext
    tnext = min(tprev + dt0, t1)
