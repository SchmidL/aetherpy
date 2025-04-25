from aetherpy.data.loader         import DEM
from aetherpy.core.multiobserver  import inverse_visibility, best_observers_from_index
from aetherpy.io.plotting         import plot_visibility_results
import numpy as np

# 1. Load DEM
dem = DEM("swissalti3d_2024_2626-1092_2_2056_5728.tif")

# 2. Define a circular target area (radius 5 cells)
nrows, ncols = dem.nrows, dem.ncols
yy, xx = np.ogrid[:nrows, :ncols]
r0, c0 = dem.index(2626500, 1092500)
radius = 10
dist = np.sqrt((yy - r0)**2 + (xx - c0)**2)
target_mask = dist <= radius

# 2a. Build a weight raster: center=1, border=0.5, linear fall-off
weights = np.zeros_like(dist, dtype=float)
mask = target_mask
weights[mask] = 10.0 - 0.5 * (dist[mask] / radius)  # at dist=0 →1.0; at dist=radius →0.5

# 3a. Inverse visibility without weighting (all targets equally important)
result_bin = inverse_visibility(
    dem,
    target_mask,
    obs_h=2.0,
    interpolation="bilinear",
    max_dist=5000.0,
    weight_by_cell=False
)
best_bin = best_observers_from_index(result_bin, k=1)[0]

# 4a. Plot raw counts vs active-observer ratio for the binary mask
plot_visibility_results(
    dem,
    result_bin,
    target_mask=target_mask,
    observer=best_bin,
    obs_metric="count",
    tgt_metric="active_ratio"
)

# 3b. Inverse visibility with per-cell weights
result_wt = inverse_visibility(
    dem,
    weights,
    obs_h=2.0,
    interpolation="bilinear",
    max_dist=5000.0,
    weight_by_cell=True
)
best_wt = best_observers_from_index(result_wt, k=1)[0]

# 4b. Plot weighted counts vs active-observer ratio
#    (overlay the same binary mask for clarity)
plot_visibility_results(
    dem,
    result_wt,
    target_mask=target_mask,
    observer=best_wt,
    obs_metric="count",
    tgt_metric="active_ratio"
)
