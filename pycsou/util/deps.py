import os
from importlib import util

cupy_enabled = util.find_spec("cupy") is not None
dask_enabled = util.find_spec("dask") is not None
jax_enabled = util.find_spec("jax") is not None