import math
from math import ceil
import numpy as np
from numba import cuda

@cuda.jit(device=True)
def _gpu_euclidean_distance(dy, dx):
    return math.hypot(dx, dy)

@cuda.jit
def _viewshed_kernel(
    arr, nrows, ncols,
    obs_r, obs_c, obs_h, maxd2,
    res_y, res_x,
    out
):
    """
    Kernel: each thread computes visibility for one cell (i,j).
    Nearest‐neighbor interpolation only.
    """
    i, j = cuda.grid(2)
    if i >= nrows or j >= ncols:
        return

    # distance filter
    dy = (i - obs_r) * res_y
    dx = (j - obs_c) * res_x
    dist2 = dy*dy + dx*dx
    if maxd2 >= 0.0 and dist2 > maxd2:
        out[i, j] = False
        return

    # end‐point heights
    height0 = arr[obs_r, obs_c] + obs_h
    height1 = arr[i, j]

    # slope to target
    dist_target = _gpu_euclidean_distance(dy, dx)
    slope_target = (height1 - height0) / dist_target if dist_target > 0.0 else -1e9

    # DDA steps
    dr = i - obs_r
    dc = j - obs_c
    steps = int(abs(dr) if abs(dr) > abs(dc) else abs(dc))
    if steps == 0:
        out[i, j] = True
        return
    d_r = dr / steps
    d_c = dc / steps

    # walk the line
    r = obs_r
    c = obs_c
    visible = True
    for _ in range(1, steps):
        r += d_r
        c += d_c
        rr = int(round(r))
        cc = int(round(c))
        hi = arr[rr, cc]
        dy_i = (r - obs_r) * res_y
        dx_i = (c - obs_c) * res_x
        dist_i = _gpu_euclidean_distance(dy_i, dx_i)
        slope_i = (hi - height0) / dist_i if dist_i > 0.0 else -1e9
        if slope_i > slope_target:
            visible = False
            break

    out[i, j] = visible

def gpu_viewshed(arr, observer, obs_h, max_dist, res_y, res_x):
    """
    Compute a nearest‐neighbor viewshed on the GPU.
    """
    nrows, ncols = arr.shape
    obs_r, obs_c = observer
    maxd2 = -1.0 if max_dist is None else float(max_dist)**2

    # Allocate device arrays
    arr_d = cuda.to_device(arr.astype(np.float32))
    out_d = cuda.device_array((nrows, ncols), dtype=np.bool_)

    # Launch kernel
    threadsperblock = (16, 16)
    blockspergrid = (ceil(nrows/16), ceil(ncols/16))
    _viewshed_kernel[blockspergrid, threadsperblock](
        arr_d, nrows, ncols,
        obs_r, obs_c, obs_h, maxd2,
        res_y, res_x,
        out_d
    )

    # Copy back
    return out_d.copy_to_host()
