![Banner](assets/banner.png)

# AetherPy

Python library for very fast and flexible terrain visibility analysis.

Supports real GeoTIFF DEMs, LOS, viewsheds (including constrained sectors, curvature correction), complex multiview observer calculation, and more. It also provides the tools for finding the optimal observer localization.

---

## 🚀 Features

- **Real GeoTIFF support**: arbitrary CRS, resolution, out‑of‑core tiling
- **High‑performance LOS & viewshed**: single and multi‑observer 
- **Constrained analysis**: range, azimuth sector, elevation angle
- **Earth‑curvature correction** *(k‑factor)*
- **Coverage metrics**: show which area can be seen from a target

---

## 📦 Installation

```bash
# From PyPI
pip install aetherpy

# Or from GitHub
pip install git+https://github.com/SchmidL/aetherpy.git

```

---

## 📝 Quickstart

```python
from aetherpy.data.loader   import DEM
from aetherpy.core          import is_visible, viewshed_sweep
from aetherpy.io.plotting   import plot_viewshed

# load a GeoTIFF DEM (or pass a NumPy array)
dem = DEM("swisssurface3d-raster_2023_2600-1199_0.5_2056_5728.tif") 

# define your observer in map coords (x, y) or pixel coords
lon, lat = 2600410.30, 1199452.00            
obs_rc   = dem.index(lon, lat) 

# quick LOS check to a target
lon2, lat2 = 2600754.88, 1199416.56
tgt_rc = dem.index(lon2, lat2)
print("Visible?", is_visible(dem, obs_rc, tgt_rc, obs_h=1.75))

# compute a 5 km viewshed at 1.75 m observer height
vs = viewshed_sweep(dem, obs_rc, obs_h=1.75, max_dist=500.0,interpolation="bilinear")

# visualize
plot_viewshed(dem, vs, observer=obs_rc, hillshade=True) 

# Save Boolean mask as GeoTIFF
save_raster("viewshed.tif",vs,dem)
```

---

## ⚖️ License

AetherPy is offered under a **dual‑license**:

1. **MIT License** – Free for:
    - Personal, academic, and non‑commercial use
    - Companies with annual revenue **≤1000000USD**
2. **Commercial License** – Required for:
    - Any organization or individual with annual revenue **>1000000USDUSD**
    - Contact us to obtain terms and pricing.

Please see LICENSE‑MIT for details of the MIT terms, or contact us for commercial licensing.

> Threshold is a guideline; please get in touch if your use case is unclear.
> 

---

## ☕ Support this project

I maintain aetherpy in my free time. If you find it valuable—especially in a commercial setting—please consider supporting my work:

[Buy me a coffee](https://www.buymeacoffee.com/schmidl) ☕

Or contact me to discuss sponsorship, custom features, or commercial collaboration:

📧 [lorenzdschmid@gmail.com](mailto:lorenzdschmid@gmail.com)

<!-- ---

## 📚 Contributing

Contributions, bug reports, and feature requests are very welcome!

Please read [CONTRIBUTING.md](https://www.notion.so/digitalearth/CONTRIBUTING.md) for details on our code of conduct and development workflow.

--- -->

## 🔗 Links

<!-- - **Homepage & Source:** [https://github.com/yourname/aetherpy](https://github.com/SchmidL/aetherpy) -->
<!-- - **Documentation:** [https://yourname.github.io/aetherpy](https://SchmidL.github.io/aetherpy) -->
- **Issue Tracker:** [https://github.com/yourname/aetherpy/issues](https://github.com/SchmidL/aetherpy/issues)

---