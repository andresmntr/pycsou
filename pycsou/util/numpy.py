import numpy as np
import numpy.typing as npt
import types

from typing import Union, Optional

from pycsou.util import deps
from pycsou.util import infer_array_module

if deps.cupy_enabled:
    import cupy as cp

if deps.dask_enabled:
    import dask.array as da

if deps.jax_enabled:
    import jax.numpy as jnp
    import jax.dlpack as jxdl

def asscalar(x: npt.ArrayLike, _xp: Optional[types.ModuleType] = None):
    if deps.is_dask:
        return x[0]
    else:
        return x.item()