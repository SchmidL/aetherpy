# aetherpy/core/gpu_multi.py

import math
import numpy as np
from numba import cuda, float32, int32, boolean

@cuda.jit(device=True)
def _gpu_euclid(dy, dx):
    return math.hypot(dx, dy)

@cuda.jit
def _inverse_kernel(
    arr, nrows, ncols,
    observer_mask,
    targets, weights, nt,
    obs_h, maxd2, res_y, res_x,
    use_bilinear,
    az1, az2, elev_min, elev_max, min_d2,
    obs_counts, tgt_counts, tgt_possible
):
    """
    One thread per target index t:
      - compute its viewshed (nearest only, for simplicity)
      - atomically increment obs_counts[i,j] by weights[t]
      - write tgt_counts[ti,tj]
    """
    t = cuda.grid(1)
    if t >= nt:
        return

    ti = targets[t, 0]
    tj = targets[t, 1]
    w  = weights[t]

    # Device‐local viewshed buffer could be huge, 
    # so we do a simple nested loop here:
    cnt = 0
    possible_ct = 0
    for i in range(nrows):
        for j in range(ncols):
            # geometric filter
            dy = (i - ti) * res_y
            dx = (j - tj) * res_x
            dist2 = dy*dy + dx*dx
            if (maxd2 >= 0 and dist2 > maxd2) or dist2 < min_d2:
                continue

            # simple nearest‐neighbor LOS (inline) 
            # you could also call a device-function for bilinear
            height0 = arr[ti, tj] + obs_h
            height1 = arr[i, j]
            d = _gpu_euclid(dy, dx)
            slope_target = (height1 - height0) / d if d>0 else -1e9

            # DDA steps as in cpu or gpu_viewshed
            dr = i - ti
            dc = j - tj
            steps = int(abs(dr) if abs(dr)>abs(dc) else abs(dc))
            vis = True
            if steps > 0:
                step_r = dr/steps
                step_c = dc/steps
                r = ti; c = tj
                for _ in range(steps):
                    r += step_r; c += step_c
                    rr = int(round(r)); cc = int(round(c))
                    hi = arr[rr, cc]
                    d_i = _gpu_euclid((r-ti)*res_y, (c-tj)*res_x)
                    slope_i = (hi - height0)/d_i if d_i>0 else -1e9
                    if slope_i > slope_target:
                        vis = False
                        break

            # every observer_mask cell passing geometry is "possible"
            if observer_mask[i, j]:
                possible_ct += 1
                # actual LOS → contribute to obs_counts
                if vis:
                    cuda.atomic.add(obs_counts, (i, j), w)
                    cnt += 1

    # write back both counts
    tgt_counts[ti, tj]    = cnt
    tgt_possible[ti, tj] = possible_ct
