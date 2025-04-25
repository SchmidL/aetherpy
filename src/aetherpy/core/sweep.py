import numpy as np
import math
from numba import njit
from .los import _is_visible, _is_visible_bilinear
from .utils import timeit

@njit
def _viewshed_naive(arr,
                    obs_r, obs_c, obs_h, maxd,
                    res_y, res_x, use_bilinear,
                    az1, az2, elev_min, elev_max, min_d,curvature_k):
    """
    Brute‑force, Numba‑accelerated viewshed with constraints:
      - use_bilinear=False → nearest‑cell LOS
      - use_bilinear=True  → bilinear LOS
      - az1,az2 in radians defining azimuth sector [az1→az2]
      - elev_min, elev_max in radians for vertical angle
      - min_d, maxd in map units for min/max range
    """
    nrows, ncols = arr.shape
    vs = np.zeros((nrows, ncols), dtype=np.bool_)
    # observer always sees itself
    vs[obs_r, obs_c] = True

    maxd2 = maxd * maxd
    min_d2 = min_d * min_d
    # height of observer
    height0 = arr[obs_r, obs_c] + obs_h

    for i in range(nrows):
        for j in range(ncols):
            # 1) distance filter
            dy = (i - obs_r) * res_y
            dx = (j - obs_c) * res_x
            dist2 = dy*dy + dx*dx
            if (maxd >= 0.0 and dist2 > maxd2) or dist2 < min_d2:
                continue

            # 2) azimuth filter
            ang = math.atan2(dy, dx)
            if ang < 0:
                ang += 2 * math.pi
            # handle wrap‑around
            if az2 >= az1:
                if not (az1 <= ang <= az2):
                    continue
            else:
                if not (ang >= az1 or ang <= az2):
                    continue

            # 3) elevation‑angle filter
            hi = arr[i, j]
            ang_v = math.atan2(hi - height0, math.sqrt(dist2))
            if ang_v < elev_min or ang_v > elev_max:
                continue

            # 4) line‑of‑sight check
            if use_bilinear:
                visible = _is_visible_bilinear(
                    arr, obs_r, obs_c, i, j, obs_h, 0.0, res_y, res_x,curvature_k
                )
            else:
                visible = _is_visible(
                    arr, obs_r, obs_c, i, j, obs_h, 0.0, res_y, res_x,curvature_k
                )

            if visible:
                vs[i, j] = True

    return vs

@timeit
def viewshed_sweep(dem, observer,
                   obs_h=0.0, max_dist=None,
                   interpolation="nearest",
                   azimuth_range=None,
                   elev_angle_range=None,
                   dist_range=None,
                   curvature_k=0.0):
    """
    Compute a constrained viewshed.

    Parameters
    ----------
    dem : DEM instance
    observer : (row, col)
    obs_h : float
        Observer height above terrain.
    max_dist : float or None
        Maximum distance (map units); None = no limit.
    interpolation : "nearest" or "bilinear"
    azimuth_range : (start_deg, end_deg) or None
    elev_angle_range : (min_deg, max_deg) or None
    dist_range : (min_dist, max_dist) or None
    """
    arr = dem.array
    r0, c0 = observer

    # 1) parse distance limits
    maxd = -1.0 if max_dist is None else float(max_dist)
    if dist_range is not None:
        min_d, maxd = dist_range
    else:
        min_d = 0.0

    # 2) parse azimuth (degrees → radians)
    if azimuth_range is not None:
        az1 = math.radians(azimuth_range[0])
        az2 = math.radians(azimuth_range[1])
    else:
        az1, az2 = 0.0, 2 * math.pi

    # 3) parse elevation angles (degrees → radians)
    if elev_angle_range is not None:
        elev_min = math.radians(elev_angle_range[0])
        elev_max = math.radians(elev_angle_range[1])
    else:
        elev_min, elev_max = -math.pi/2, math.pi/2

    # 4) interpolation mode
    use_bi = (interpolation.lower() == "bilinear")

    return _viewshed_naive(
        arr,
        r0, c0, obs_h, maxd,
        dem.res_y, dem.res_x, use_bi,
        az1, az2, elev_min, elev_max, min_d,curvature_k=0.0
    )
