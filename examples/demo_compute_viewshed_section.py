from aetherpy.data.loader   import DEM
from aetherpy.core          import is_visible, viewshed_sweep
from aetherpy.io.plotting   import plot_viewshed

# load a GeoTIFF DEM (or pass a NumPy array)
dem = DEM("swissalti3d_2024_2626-1092_2_2056_5728.tif")

# define your observer in map coords (x, y) or pixel coords
lon, lat = 2626572.40, 1092492.59            
obs_rc   = dem.index(lon, lat)        # -> (row, col)

# quick LOS check to a target
lon2, lat2 = 2626448.06, 1092380.26
tgt_rc = dem.index(lon2, lat2)
print("Visible?", is_visible(dem, obs_rc, tgt_rc, obs_h=1.75))

# compute a 5 km viewshed at 1.75 m observer height
vs = viewshed_sweep(
    dem,
    obs_rc,
    obs_h=1.75,
    interpolation="bilinear",
    max_dist=5000,
    dist_range=(25,250),
    #azimuth_range=(45,135),
    #elev_angle_range=(0,30)
)

# visualize
plot_viewshed(dem, vs, observer=obs_rc)	
