# aetherpy/core/multiobserver.py

import numpy as np
import math
from collections import namedtuple
from .sweep import viewshed_sweep, _viewshed_naive
from numba import njit, prange

# Namedtuple to carry both counts and ratios
VisibilityResult = namedtuple(
    "VisibilityResult",
    [
        "obs_counts",           # raw # targets seen per observer
        "obs_ratio",            # obs_counts / total_targets
        "tgt_counts",           # raw # observers that see each target
        "tgt_ratio",            # tgt_counts / total_observers
        "tgt_possible_counts",  # raw # possible observers per target
        "tgt_possible_ratio",   # tgt_counts / tgt_possible_counts
        "tgt_active_ratio",     # tgt_counts / # active_observers
    ]
)

def inverse_visibility(
    dem,
    target_mask,
    obs_h=0.0,
    interpolation="nearest",
    max_dist=None,
    azimuth_range=None,
    elev_angle_range=None,
    dist_range=None,
    observer_mask=None,
    n_jobs=None,            # kept for signature compatibility
):
    """
    Inverse‐viewshed computation using a Numba‐parallel backend.

    Returns a VisibilityResult with:
      - obs_counts:  raw count of target cells seen by each observer
      - obs_ratio:   obs_counts / total_targets
      - tgt_counts:  raw count of observers that see each target cell
      - tgt_ratio:   tgt_counts / total_observers

    Parameters
    ----------
    dem : DEM
    target_mask : 2D bool array
        True for cells in your target area (e.g. a lake).
    obs_h : float
        Observer height above terrain.
    interpolation : "nearest" or "bilinear"
    max_dist : float or None
    azimuth_range : (start_deg, end_deg) or None
    elev_angle_range : (min_deg, max_deg) or None
    dist_range : (min_dist, max_dist) or None
    observer_mask : 2D bool array, optional
        Which cells are allowed as observers. If None, all cells are allowed.
    n_jobs : ignored (parallelism controlled by Numba)
    """
    nrows, ncols = dem.array.shape

    # gather target indices for Numba
    targets = np.array(np.argwhere(target_mask), dtype=np.int32)
    total_targets = targets.shape[0]
    if total_targets == 0:
        raise ValueError("target_mask contains no True cells")

    # define observers
    if observer_mask is None:
        observer_mask = np.ones_like(target_mask, dtype=bool)
    total_observers = int(observer_mask.sum())

    # parse distance limits
    maxd = -1.0 if max_dist is None else float(max_dist)
    if dist_range is not None:
        min_d, maxd = dist_range
    else:
        min_d = 0.0

    # parse azimuth_range (deg → rad)
    if azimuth_range is not None:
        az1 = math.radians(azimuth_range[0])
        az2 = math.radians(azimuth_range[1])
    else:
        az1, az2 = 0.0, 2 * math.pi

    # parse elevation_angle_range (deg → rad)
    if elev_angle_range is not None:
        elev_min = math.radians(elev_angle_range[0])
        elev_max = math.radians(elev_angle_range[1])
    else:
        elev_min, elev_max = -math.pi/2, math.pi/2

    # interpolation flag
    use_bi = (interpolation.lower() == "bilinear")

    # call the Numba‐parallel inverse‐viewshed
    obs_counts, tgt_counts, tgt_possible = _inverse_counts_jit(
        dem.array,
        targets,
        observer_mask,
        obs_h,
        maxd,
        dem.res_y, dem.res_x,
        use_bi,
        az1, az2,
        elev_min, elev_max,
        min_d
    )

    # compute ratios
    obs_ratio = obs_counts.astype(float) / float(total_targets)
    # 1) target‐side practical ratio
    if total_observers > 0:
        tgt_ratio = tgt_counts.astype(float) / float(total_observers)
    else:
        tgt_ratio = np.zeros_like(obs_ratio)

    # 2) target‐side theoretical ratio
    tgt_possible_ratio = np.zeros_like(tgt_ratio)
    mask_pos = (tgt_possible > 0)
    tgt_possible_ratio[mask_pos] = (
        tgt_counts[mask_pos].astype(float) / tgt_possible[mask_pos].astype(float)
    )

    # 3) target‐side practical (active observers) ratio
    active_obs = np.sum(obs_counts > 0)
    tgt_active_ratio = np.zeros_like(tgt_ratio)
    if active_obs > 0:
        tgt_active_ratio = tgt_counts.astype(float) / float(active_obs)

    return VisibilityResult(
        obs_counts, obs_ratio,
        tgt_counts, tgt_ratio,
        tgt_possible, tgt_possible_ratio,
        tgt_active_ratio
    )


def best_observers_from_index(results, k=1):
    """
    Return the top-k observer locations for a given VisibilityResult.
    """
    flat = results.obs_counts.flatten()
    idx = np.argpartition(-flat, k-1)[:k]
    rows, cols = np.unravel_index(idx, results.obs_counts.shape)
    return list(zip(rows, cols))


@njit(parallel=True)
def _inverse_counts_jit(arr, targets, observer_mask,
                        obs_h, maxd, res_y, res_x,
                        use_bilinear,
                        az1, az2, elev_min, elev_max,
                        min_d):
    """
    Numba‐parallel inverse‐viewshed helper.

    Loops over targets in parallel, computes each target’s viewshed
    via _viewshed_naive, and accumulates observer and target counts.
    """
    nrows, ncols = arr.shape
    nt = targets.shape[0]

    obs_counts = np.zeros((nrows, ncols), np.int32)
    tgt_counts = np.zeros((nrows, ncols), np.int32)
    tgt_possible = np.zeros((nrows, ncols), np.int32)

    for t in prange(nt):
        ti = targets[t, 0]
        tj = targets[t, 1]

        # count theoretical possible observers (geometry only)
        possible_ct = 0

        # compute viewshed from this target
        vs = _viewshed_naive(
            arr,
            ti, tj, obs_h, maxd,
            res_y, res_x, use_bilinear,
            az1, az2, elev_min, elev_max,
            min_d
        )

        # accumulate counts
        cnt = 0
        for i in range(nrows):
            for j in range(ncols):
                # check geometric constraints _first_ (we know vs implies constraints)
                dy = (i - ti) * res_y
                dx = (j - tj) * res_x
                dist2 = dy*dy + dx*dx
                if (maxd >= 0.0 and dist2 > maxd*maxd) or dist2 < min_d*min_d:
                    continue
                ang = math.atan2(dy, dx)
                if ang < 0:
                    ang += 2*math.pi
                if az2 >= az1:
                    if not (az1 <= ang <= az2):
                        continue
                else:
                    if not (ang >= az1 or ang <= az2):
                        continue
                hi = arr[i, j]
                ang_v = math.atan2(hi - (arr[ti, tj] + obs_h), math.sqrt(dist2))
                if ang_v < elev_min or ang_v > elev_max:
                    continue
                # observer is theoretically possible
                if observer_mask[i, j]:
                    possible_ct += 1
                # now count actual LOS
                if vs[i, j] and observer_mask[i, j]:
                    obs_counts[i, j] += 1
                    cnt += 1
        tgt_counts[ti, tj] = cnt
        tgt_possible[ti, tj] = possible_ct

    return obs_counts, tgt_counts, tgt_possible
