import numpy as np
from typing import Callable
from warnings import warn

from pycsou.util import cupy_enabled, dask_enabled, jax_enabled

if cupy_enabled:
    import cupy as cp

if dask_enabled:
    import dask.array as da

if jax_enabled:
    import jax.numpy as jnp
    import jax.dlpack as jxdl


def infer_array_module(decorated_object_type='method'):
    def make_decorator(call_fun: Callable):
        def wrapper(*args, **kwargs):
            if decorated_object_type == 'method':
                arr = args[1] #First argument is self
            else:
                arr = args[0]
            if cupy_enabled and isinstance(arr, cp.ndarray):
                xp = cp
            elif dask_enabled and isinstance(arr, da.core.Array):
                xp = da
            elif isinstance(arr, np.ndarray):
                xp = np
            elif jax_enabled and isinstance(arr, jnp.ndarray):
                xp = jnp
            else:
                warn('Unknown array module. Falling back to Numpy backend')
                xp = np
            kwargs['_xp'] = xp
            return call_fun(*args, **kwargs)
        return wrapper
    return make_decorator
