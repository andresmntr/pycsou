import numpy as np
from typing import Callable
from warnings import warn
from numbers import Number

from pycsou.util import deps

if deps.cupy_enabled:
    import cupy as cp

if deps.dask_enabled:
    import dask.array as da

if deps.jax_enabled:
    import jax.numpy as jnp
    import jax.dlpack as jxdl


def infer_array_module(decorated_object_type='method'):
    def make_decorator(call_fun: Callable):
        def wrapper(*args, **kwargs):
            if decorated_object_type == 'method':
                arr = args[1] #First argument is self
            else:
                arr = args[0]
            xp = infer_module_from_array(arr)
            kwargs['_xp'] = xp
            return call_fun(*args, **kwargs)
        return wrapper
    return make_decorator

def infer_module_from_array(arr):
    if deps.cupy_enabled and isinstance(arr, cp.ndarray):
        xp = cp
    elif deps.dask_enabled and isinstance(arr, da.core.Array):
        xp = da
    elif isinstance(arr, np.ndarray):
        xp = np
    elif deps.jax_enabled and isinstance(arr, jnp.ndarray):
        xp = jnp
    else:
        if not isinstance(arr, Number):
            warn('Unknown array module. Falling back to Numpy backend')
        xp = np
    return xp

def to_cupy_conditional(x, y):
    """Convert y to cupy array conditional to x being a cupy array or viceversa
    Parameters
    ----------
    x : :obj:`numpy.ndarray` or :obj:`cupy.ndarray`
        Array to evaluate
    y : :obj:`numpy.ndarray` or :obj:`cupy.ndarray`
        Array to convert
    Returns
    -------
    y : :obj:`cupy.ndarray`
        Converted array
    """
    if deps.cupy_enabled:
        if cp.get_array_module(x) == cp and cp.get_array_module(y) == np:
            y = cp.asarray(y)
        elif cp.get_array_module(y) == cp and cp.get_array_module(x) == np:
            x = cp.asarray(x)
    return x, y

def to_dask_conditional(x, y):
    """Convert y to dask array conditional to x being a dask array or viceversa
    Parameters
    ----------
    x : :obj:`numpy.ndarray` or :obj:`dask.array.core.Array`
        Array to evaluate
    y : :obj:`numpy.ndarray` or :obj:`dask.array.core.Array`
        Array to convert
    Returns
    -------
    y : :obj:`cupy.ndarray`
        Converted array
    """
    if deps.is_dask(x) and deps.is_numpy(y):
        y = da.from_array(y)
    elif deps.is_dask(y) and deps.is_numpy(x):
        x = da.from_array(x)
    return x, y

def to_jax_conditional(x, y):
    """Convert y to jax array conditional to x being a jax array or viceversa
    Parameters
    ----------
    x : :obj:`numpy.ndarray` or :obj:`jax.numpy.ndarray`
        Array to evaluate
    y : :obj:`numpy.ndarray` or :obj:`jax.numpy.ndarray`
        Array to convert
    Returns
    -------
    y : :obj:`jax.ndarray`
        Converted array
    """
    if deps.is_jax(x) and deps.is_numpy(y):
        y = jnp.asarray(y)
    elif deps.is_jax(y) and deps.is_numpy(x):
        x = jnp.asarray(x)
    return x, y

def to_xp_conditional(x,y):
    if deps.is_numpy(x) or deps.is_numpy(y):
        x, y = to_cupy_conditional(x,y)
        x, y = to_dask_conditional(x,y)
        x, y = to_jax_conditional(x,y)
    else:
        if infer_module_from_array(x) != infer_module_from_array(y):
            raise TypeError("Invalid module types. Not compatible.")

    return x, y