#fft, radon

import numpy as np
import types
from typing import Optional
from pycsou.util import deps, infer_array_module
from warnings import warn

if deps.cupy_enabled:
    import cupy as cp
if deps.dask_enabled:
    import dask.array as da
if deps.jax_enabled:
    import jax.numpy as jnp
    from jax import jacfwd
    import jax.dlpack as jxdl

class FFTOp(object):
    @infer_array_module(decorated_object_type='method')
    def __call__(self, x, _xp: Optional[types.ModuleType] = None):
        return _xp.fft.fft(x)

    @infer_array_module(decorated_object_type='method')
    def jacobian(self, x, _xp: Optional[types.ModuleType] = None):
        if _xp == cp:
            arr = jxdl.from_dlpack(x.astype(_xp.float32).toDlpack()) # Zero-copy conversion from Cupy to JAX arrays only works with float32 dtypes. 
            warn('Automatic differentiation with Cupy arrays only works with float32 precision.')
        elif _xp == da:
            raise NotImplementedError('Automatic differentiation does not support with lazy Dask arrays.')
        else:
            arr = jnp.asarray(x)
        jaxobian_eval = jacfwd(self.__call__)(arr)
        return _xp.asarray(jaxobian_eval)