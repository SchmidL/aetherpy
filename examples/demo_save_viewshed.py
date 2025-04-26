from aetherpy.data.loader   import DEM
from aetherpy.core          import is_visible, viewshed_sweep
from aetherpy.io.raster     import save_raster


# load a GeoTIFF DEM (or pass a NumPy array)
dem = DEM("swisssurface3d-raster_2023_2600-1199_0.5_2056_5728.tif") 

# define your observer in map coords (x, y) or pixel coords
lon, lat = 2600410.30, 1199452.00            
obs_rc   = dem.index(lon, lat) 

# compute a 5 km viewshed at 1.75 m observer height
vs = viewshed_sweep(dem, obs_rc, obs_h=1.75, max_dist=500.0,interpolation="bilinear")


# Save Boolean mask as uint8 GeoTIFF
save_raster(
    "viewshed.tif",
    vs,
    dem
)