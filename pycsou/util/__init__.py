from pycsou.util.misc import is_range_broadcastable, range_broadcast_shape, peaks
from pycsou.util.stats import P2Algorithm
from pycsou.util.deps import cupy_enabled, dask_enabled, jax_enabled
from pycsou.util.backend import infer_array_module, infer_module_from_array
from pycsou.util.numpy import asscalar
from pycsou.util.skimage import block_reduce