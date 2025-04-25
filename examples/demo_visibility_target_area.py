from aetherpy.data.loader      import DEM
from aetherpy.core.multiobserver import inverse_visibility
from aetherpy.io.plotting      import plot_visibility_results

from aetherpy.core.multiobserver import best_observers_from_index

# 1. Load DEM
dem = DEM("swissalti3d_2024_2626-1092_2_2056_5728.tif")

# 2. Define target area (e.g., a lake) as a boolean mask
#    Here we’ll just create a dummy circular mask around some center—
#    replace with your real rasterized polygon.
import numpy as np
nrows, ncols = dem.nrows, dem.ncols
yy, xx = np.ogrid[:nrows, :ncols]
center = dem.index(2626500, 1092500)
r0, c0 = center
radius = 5  # cells
target_mask = ((yy - r0)**2 + (xx - c0)**2) <= radius**2

# 3. Compute inverse visibility indices
result = inverse_visibility(dem,
    target_mask=target_mask,
    obs_h=2.0,
    interpolation="bilinear",
    max_dist=5000.0,
    azimuth_range=None,
    elev_angle_range=None,
    dist_range=None,
)

# 4. Pick an example observer (e.g. the best one by index)
best_obs = best_observers_from_index(result, k=10)[0]

# 5. Plot both maps
# plot raw counts
plot_visibility_results(
    dem,
    result,
    target_mask=target_mask,
    observer=best_obs,
    obs_metric="count",
    tgt_metric="active_ratio",
)
