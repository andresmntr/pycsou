import numpy as np
import types

from typing import Union, Optional

from pycsou.util import infer_array_module, cupy_enabled, dask_enabled, jax_enabled

if cupy_enabled:
    import cupy as cp

if dask_enabled:
    import dask.array as da

if jax_enabled:
    import jax.numpy as jnp
    import jax.dlpack as jxdl

@infer_array_module(decorated_object_type='method')
def asscalar(x: Union[np.ndarray, cp.ndarray, da.core.Array, jnp.ndarray], _xp: Optional[types.ModuleType] = None):
    if dask_enabled and _xp == da.core.Array:
        return x[0]
    else:
        return x.item()