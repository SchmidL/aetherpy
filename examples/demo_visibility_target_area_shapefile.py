from aetherpy.data.loader      import DEM
from aetherpy.core.multiobserver import inverse_visibility
from aetherpy.io.plotting      import plot_visibility_results

from aetherpy.core.multiobserver import best_observers_from_index

# 1. Load DEM
dem = DEM("swissalti3d_2024_2626-1092_2_2056_5728.tif")

# 2. Define target area (e.g., a lake) as a boolean mask
target_mask = dem.rasterize_mask(
    "waterbody.shp",
    all_touched=True
)


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

