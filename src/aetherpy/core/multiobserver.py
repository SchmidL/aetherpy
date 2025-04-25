# aetherpy/core/multiobserver.py

import numpy as np
import math
from collections import namedtuple
from .sweep import viewshed_sweep, _viewshed_naive
from numba import njit, prange

# Namedtuple to carry counts + all three ratio types
VisibilityResult = namedtuple(
    "VisibilityResult",
    [
        "obs_counts",            # raw weighted sum of target importance seen per observer
        "obs_ratio",             # obs_counts / total_weight
        "tgt_counts",            # raw # observers that see each target
        "tgt_ratio",             # tgt_counts / total_observers
        "tgt_possible_counts",   # raw # possible observers per target
        "tgt_possible_ratio",    # tgt_counts / tgt_possible_counts
        "tgt_active_ratio",      # tgt_counts / # active_observers
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
    weight_by_cell=False,   # if True, interpret target_mask values (0–1) as weights
    n_jobs=None,            # ignored; parallelism via Numba
):
    """
    Inverse‐viewshed computation with optional per‐cell weighting.

    Returns a VisibilityResult containing:
      - obs_counts:          weighted sum of target importance seen per observer
      - obs_ratio:           obs_counts / total_weight
      - tgt_counts:          count of observers that see each target
      - tgt_ratio:           tgt_counts / total_observers
      - tgt_possible_counts: count of all *possible* observers per target
      - tgt_possible_ratio:  tgt_counts / tgt_possible_counts
      - tgt_active_ratio:    tgt_counts / (# observers that saw ≥1 target)

    Parameters
    ----------
    dem : DEM
    target_mask : 2D array
        If weight_by_cell=False: bool mask of target cells.
        If weight_by_cell=True: float mask of weights [0..1].
    obs_h : float
        Observer height above terrain.
    interpolation : "nearest" or "bilinear"
    max_dist : float or None
        Maximum distance in map units.
    azimuth_range : (start_deg, end_deg) or None
    elev_angle_range : (min_deg, max_deg) or None
    dist_range : (min_dist, max_dist) or None
    observer_mask : 2D bool array, optional
        Which cells may act as observers.
    weight_by_cell : bool
        If True, use target_mask values as weights; else treat mask as boolean.
    """
    nrows, ncols = dem.array.shape

    # Build weights array
    arr_w = (
        target_mask.astype(np.float64)
        if weight_by_cell
        else (target_mask != 0).astype(np.float64)
    )
    # Extract nonzero‐weight targets
    idxs = np.argwhere(arr_w > 0.0)
    targets = np.array(idxs, dtype=np.int32)
    weights = np.array([arr_w[i, j] for i, j in idxs], dtype=np.float64)
    total_weight = weights.sum()
    if total_weight <= 0.0:
        raise ValueError("Sum of target weights must be > 0")

    # Observer mask
    if observer_mask is None:
        observer_mask = np.ones_like(arr_w, dtype=bool)
    total_observers = int(observer_mask.sum())

    # Parse distance limits
    maxd = -1.0 if max_dist is None else float(max_dist)
    if dist_range is not None:
        min_d, maxd = dist_range
    else:
        min_d = 0.0

    # Parse azimuth (degrees → radians)
    if azimuth_range is not None:
        az1 = math.radians(azimuth_range[0])
        az2 = math.radians(azimuth_range[1])
    else:
        az1, az2 = 0.0, 2 * math.pi

    # Parse elevation angles (degrees → radians)
    if elev_angle_range is not None:
        elev_min = math.radians(elev_angle_range[0])
        elev_max = math.radians(elev_angle_range[1])
    else:
        elev_min, elev_max = -math.pi / 2, math.pi / 2

    # Interpolation mode
    use_bi = (interpolation.lower() == "bilinear")

    # Run Numba‐parallel inverse‐viewshed kernel
    obs_counts, tgt_counts, tgt_possible = _inverse_counts_jit(
        dem.array,
        targets,
        weights,
        observer_mask,
        obs_h,
        maxd,
        dem.res_y, dem.res_x,
        use_bi,
        az1, az2,
        elev_min, elev_max,
        min_d
    )

    # Compute ratios
    obs_ratio = obs_counts / total_weight
    tgt_ratio = (
        tgt_counts.astype(float) / total_observers
        if total_observers > 0
        else np.zeros_like(obs_counts)
    )

    # Theoretical ratio per target
    tgt_possible_ratio = np.zeros_like(tgt_ratio)
    mask_pos = (tgt_possible > 0)
    tgt_possible_ratio[mask_pos] = (
        tgt_counts[mask_pos].astype(float) / tgt_possible[mask_pos].astype(float)
    )

    # Active observers ratio per target
    active_obs = int((obs_counts > 0).sum())
    tgt_active_ratio = (
        tgt_counts.astype(float) / active_obs
        if active_obs > 0
        else np.zeros_like(tgt_counts, dtype=float)
    )

    return VisibilityResult(
        obs_counts,
        obs_ratio,
        tgt_counts,
        tgt_ratio,
        tgt_possible,
        tgt_possible_ratio,
        tgt_active_ratio
    )


def best_observers_from_index(results, k=1):
    """
    Return the top-k observer locations for a given VisibilityResult.
    """
    flat = results.obs_counts.flatten()
    idx = np.argpartition(-flat, k - 1)[:k]
    rows, cols = np.unravel_index(idx, results.obs_counts.shape)
    return list(zip(rows, cols))


@njit(parallel=True)
def _inverse_counts_jit(
    arr, targets, weights, observer_mask,
    obs_h, maxd, res_y, res_x,
    use_bilinear,
    az1, az2, elev_min, elev_max,
    min_d
):
    """
    Numba‐parallel inverse‐viewshed helper.
    """
    nrows, ncols = arr.shape
    nt = targets.shape[0]

    obs_counts = np.zeros((nrows, ncols), np.float64)
    tgt_counts = np.zeros((nrows, ncols), np.int32)
    tgt_possible = np.zeros((nrows, ncols), np.int32)

    for t in prange(nt):
        ti = targets[t, 0]
        tj = targets[t, 1]
        w = weights[t]

        # compute the viewshed from this target
        vs = _viewshed_naive(
            arr,
            ti, tj, obs_h, maxd,
            res_y, res_x, use_bilinear,
            az1, az2, elev_min, elev_max,
            min_d
        )

        # accumulate counts
        cnt = 0
        possible_ct = 0
        for i in range(nrows):
            for j in range(ncols):
                # geometric constraints (same as in sweep)
                dy = (i - ti) * res_y
                dx = (j - tj) * res_x
                dist2 = dy*dy + dx*dx
                if (maxd >= 0.0 and dist2 > maxd*maxd) or dist2 < min_d*min_d:
                    continue
                ang = math.atan2(dy, dx)
                if ang < 0:
                    ang += 2 * math.pi
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
                # possible observer
                if observer_mask[i, j]:
                    possible_ct += 1
                # actual visibility
                if vs[i, j] and observer_mask[i, j]:
                    obs_counts[i, j] += w
                    cnt += 1

        tgt_counts[ti, tj] = cnt
        tgt_possible[ti, tj] = possible_ct

    return obs_counts, tgt_counts, tgt_possible
