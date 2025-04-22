import numpy as np
from numba import njit
from .los import _is_visible, _is_visible_bilinear
from .utils import timeit

@njit
def _viewshed_naive(arr, obs_r, obs_c, obs_h, max_dist,
                    res_y, res_x, use_bilinear):
    """
    Brute‑force viewshed:
      use_bilinear=False → calls _is_visible (nearest)
      use_bilinear=True  → calls _is_visible_bilinear
    """
    nrows, ncols = arr.shape
    vs = np.zeros((nrows, ncols), dtype=np.bool_)
    vs[obs_r, obs_c] = True
    maxd2 = max_dist * max_dist

    for i in range(nrows):
        for j in range(ncols):
            # skip outside radius if limited
            if max_dist >= 0.0:
                dy = (i - obs_r) * res_y
                dx = (j - obs_c) * res_x
                if dy*dy + dx*dx > maxd2:
                    continue

            # choose the appropriate LOS kernel
            visible = ( _is_visible_bilinear(arr, obs_r, obs_c,
                                             i, j, obs_h, 0.0,
                                             res_y, res_x)
                        if use_bilinear
                        else
                        _is_visible(arr, obs_r, obs_c,
                                    i, j, obs_h, 0.0,
                                    res_y, res_x) )
            if visible:
                vs[i, j] = True

    return vs

@timeit
def viewshed_sweep(dem, observer, obs_h=0.0, max_dist=None, interpolation="nearest"):
    """
    Public API:
      dem: DEM instance
      observer: (row, col)
      obs_h: observer height
      max_dist: maximum radius in map units (None = no limit)
      interpolation: "nearest" (default) or "bilinear"

    Returns a boolean array of the same shape as dem.array.
    """
    arr = dem.array
    r0, c0 = observer
    maxd = -1.0 if max_dist is None else float(max_dist)
    
    use_bi = (interpolation.lower() == "bilinear")
    return _viewshed_naive(arr, r0, c0, obs_h, maxd,
                           dem.res_y, dem.res_x,
                           use_bi)
