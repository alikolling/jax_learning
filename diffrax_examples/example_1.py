import jax
import jax.numpy as jnp
from diffrax import (
    diffeqsolve,
    Dopri5,
    ODETerm,
    SaveAt,
    PIDController,
)

vector_field = lambda t, y, args: -y
term = ODETerm(vector_field)
solver = Dopri5()
saveat = SaveAt(ts=[0.0, 1.0, 2.0, 3.0])
step_size_controller = PIDController(rtol=1e-5, atol=1e-5)

sol = diffeqsolve(
    term,
    solver,
    t0=0,
    t1=3,
    dt0=0.1,
    y0=1,
    saveat=saveat,
    stepsize_controller=step_size_controller,
)

print(sol.ts)
print(sol.ys)
