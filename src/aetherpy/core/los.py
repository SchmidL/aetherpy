import math
from numba import njit
from .utils import compute_slope, euclidean_distance

@njit
def _is_visible(arr, r0, c0, r1, c1, obs_h, tgt_h, res_y, res_x, curvature_k):
    """
    Numba‑accelerated LOS on raw elevation array with real‑world scaling.
    """
    height0 = arr[r0, c0] + obs_h
    height1 = arr[r1, c1] + tgt_h
    # — curvature correction at the target —
    if curvature_k > 0.0:
        d_t = euclidean_distance(r0, c0, r1, c1, res_y, res_x)
        drop_t = (d_t * d_t) / (2.0 * 6371000.0 * curvature_k)
    else:
        drop_t = 0.0
    slope_target = compute_slope((height1 - height0) - drop_t,
                                 r0, c0, r1, c1, res_y, res_x)

    # Bresenham’s setup (steep vs shallow)
    dr = abs(r1 - r0)
    dc = abs(c1 - c0)
    steep = dr > dc
    if steep:
        r0_, c0_ = c0, r0
        r1_, c1_ = c1, r1
    else:
        r0_, c0_ = r0, c0
        r1_, c1_ = r1, c1

    dx = abs(c1_ - c0_)
    dy = abs(r1_ - r0_)
    error = dx // 2
    ystep = 1 if r1_ > r0_ else -1
    xstep = 1 if c1_ > c0_ else -1
    y = r0_
    x = c0_ + xstep

    # step along the line (excluding endpoints)
    while x != c1_:
        error -= dy
        if error < 0:
            y += ystep
            error += dx

        # map back to row/col
        rr, cc = (x, y) if steep else (y, x)

        # if any intermediate slope exceeds the target slope → blocked
        # — curvature correction at the intermediate point —
        if curvature_k > 0.0:
            d_i = euclidean_distance(r0, c0, rr, cc, res_y, res_x)
            drop_i = (d_i * d_i) / (2.0 * 6371000.0 * curvature_k)
        else:
            drop_i = 0.0
        slope_i = compute_slope((arr[rr, cc] - height0) - drop_i,
                                r0, c0, rr, cc, res_y, res_x)
        if slope_i > slope_target:
            return False

        x += xstep

    return True

def is_visible(dem, p1, p2, obs_h=0.0, tgt_h=0.0, interpolation="nearest", curvature_k=0.0):
    """
    Public API:
      dem  – DEM instance
      p1,p2: (row, col) tuples in the DEM grid
      obs_h, tgt_h: height offsets above the surface
      interpolation: "nearest" or "bilinear"
    Returns True if p2 is visible from p1.
    """
    r0, c0 = p1
    r1, c1 = p2
    arr = dem.array
    ry, rx = dem.res_y, dem.res_x
    if interpolation == "nearest":
        return _is_visible(arr, r0, c0, r1, c1, obs_h, tgt_h, ry, rx, curvature_k)
    elif interpolation == "bilinear":
        return _is_visible_bilinear(arr, r0, c0, r1, c1, obs_h, tgt_h, ry, rx,                       curvature_k)
    else:
        raise ValueError(f"Unknown interpolation {interpolation!r}")

@njit
def _bilinear_sample(arr, row_f, col_f):
    """
    Numba‑friendly bilinear interpolation on a 2D array.
    """
    nrows, ncols = arr.shape
    r0 = int(math.floor(row_f))
    c0 = int(math.floor(col_f))
    dr = row_f - r0
    dc = col_f - c0

    # clamp indices
    r1 = r0 + 1 if r0 + 1 < nrows else r0
    c1 = c0 + 1 if c0 + 1 < ncols else c0

    h00 = arr[r0, c0]
    h10 = arr[r1, c0]
    h01 = arr[r0, c1]
    h11 = arr[r1, c1]

    return (h00 * (1 - dr) * (1 - dc)
          + h10 * dr       * (1 - dc)
          + h01 * (1 - dr) * dc
          + h11 * dr       * dc)

@njit
def _is_visible_bilinear(arr, r0, c0, r1, c1, obs_h, tgt_h, res_y, res_x, curvature_k):
    """
    Parametric (DDA) LOS with bilinear sampling.
    Oversamples the line at roughly one sample per cell in the longest axis.
    """
    # end‐point heights
    h0 = _bilinear_sample(arr, r0, c0) + obs_h
    h1 = _bilinear_sample(arr, r1, c1) + tgt_h

    # number of steps = length in grid‐units of the largest axis
    dr = r1 - r0
    dc = c1 - c0
    n_steps = int(math.ceil(max(abs(dr), abs(dc))))
    if n_steps == 0:
        return True

    # parametric increments
    d_r = dr / n_steps
    d_c = dc / n_steps

    # slope to target with curvature drop
    dist_target = euclidean_distance(r0, c0, r1, c1, res_y, res_x)
    if curvature_k > 0.0:
        drop_t = (dist_target * dist_target) / (2.0 * 6371000.0 * curvature_k)
    else:
        drop_t = 0.0
    slope_target = ((h1 - h0) - drop_t) / dist_target if dist_target > 0.0 else -1e9

    for step in range(1, n_steps):
        rf = r0 + d_r * step
        cf = c0 + d_c * step
        hi = _bilinear_sample(arr, rf, cf)
        # distance so far
        dist_i = euclidean_distance(r0, c0, rf, cf, res_y, res_x)
        # apply curvature drop at this sample
        if curvature_k > 0.0:
            drop_i = (dist_i * dist_i) / (2.0 * 6371000.0 * curvature_k)
        else:
            drop_i = 0.0
        slope_i = ((hi - h0) - drop_i) / dist_i if dist_i > 0.0 else -1e9
        
        if slope_i > slope_target:
            return False

    return True
