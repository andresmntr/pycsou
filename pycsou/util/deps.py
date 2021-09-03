import numpy as np
from importlib import util

cupy_enabled = util.find_spec("cupy") is not None
dask_enabled = util.find_spec("dask") is not None
jax_enabled = util.find_spec("jax") is not None

if cupy_enabled:
    import cupy as cp

if dask_enabled:
    import dask.array as da

if jax_enabled:
    import jax.numpy as jnp

is_cupy = lambda xp: (cupy_enabled and isinstance(xp, cp.ndarray))
is_dask = lambda xp: (dask_enabled and isinstance(xp, da.core.Array))
is_jax = lambda xp: (jax_enabled and isinstance(xp, jnp.ndarray))
is_numpy = lambda xp: isinstance(xp, np.ndarray)