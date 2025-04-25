# aetherpy/io/plotting.py
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LightSource, ListedColormap

def plot_viewshed(
     dem,
     vs,
     observer=None,
     hillshade=False,
     dem_cmap='gray',
     vs_color='red',
     alpha=0.4,
     figsize=(8, 6)
 ):
     """
     Plot the DEM and overlay a viewshed mask, with optional hillshade.

     Parameters
     ----------
     dem : DEM instance
         Must have `.array`, `.nrows`, `.ncols`, and, if georeferenced,
         `.transform` & `.coord()` available.
     vs : 2D bool array
         Output of viewshed_sweep (True = visible).
     observer : tuple (row, col), optional
         Mark this point with an 'x'.
     hillshade : bool, default False
         If True, render the DEM as a hillshade instead of raw elevations.
     dem_cmap : str or Colormap, default 'gray'
         Colormap for the DEM or hillshade.
     vs_color : str or tuple, default 'red'
         Color for the viewshed overlay.
     alpha : float, default 0.4
         Transparency for the viewshed overlay.
     figsize : tuple, optional
         Figure size.
     """
     fig, ax = plt.subplots(figsize=figsize)

     # Determine plotting extent
     if dem.transform is not None:
         left, top = dem.transform * (0, 0)
         right, bottom = dem.transform * (dem.ncols, dem.nrows)
         extent = (left, right, bottom, top)
     else:
         extent = None

     # Plot DEM or hillshade
     if hillshade:
         ls = LightSource(azdeg=315, altdeg=45)
         shade = ls.hillshade(
             dem.array,
             vert_exag=1,
             dx=dem.res_x if dem.transform else 1,
             dy=dem.res_y if dem.transform else 1
         )
         ax.imshow(shade, origin='upper', extent=extent, cmap=dem_cmap)
     else:
         ax.imshow(dem.array, origin='upper', extent=extent, cmap=dem_cmap)

     # Overlay a solid-color viewshed mask
     mask = np.where(vs, 1, np.nan)
     cmap = ListedColormap([vs_color])
     ax.imshow(
         mask,
         origin='upper',
         extent=extent,
         cmap=cmap,
         vmin=1,
         vmax=1,
         alpha=alpha
     )

     # Mark observer location
     if observer is not None:
         if dem.transform is not None:
             x, y = dem.coord(*observer)
             ax.plot(x, y, marker='x', color='white', markersize=10)
         else:
             ax.plot(observer[1], observer[0], marker='x', color='white', markersize=10)

     ax.set_xlabel('X')
     ax.set_ylabel('Y')
     plt.show()



def plot_visibility_results(
    dem,
    result,
    target_mask=None,
    observer=None,
    obs_metric="ratio",
    tgt_metric="ratio",
    figsize=(14, 6),
):
    """
    Plot two panels: observer metric (left) and target metric (right).

    Parameters
    ----------
    dem : DEM
    result : VisibilityResult
        Namedtuple from inverse_visibility.
    target_mask : 2D bool array, optional
        To overlay as red on the target plot.
    observer : (row, col), optional
        Mark this observer on the left plot.
    obs_metric : str
        One of:
          - "count"          -> result.obs_counts
          - "ratio"          -> result.obs_ratio
    tgt_metric : str
        One of:
          - "count"          -> result.tgt_counts
          - "ratio"          -> result.tgt_ratio
          - "possible_count" -> result.tgt_possible_counts
          - "possible_ratio" -> result.tgt_possible_ratio
          - "active_ratio"   -> result.tgt_active_ratio
    figsize : tuple
        Figure size.
    """
    # Map metric names to (label, data array)
    obs_map = {
        "count": ("Observer visible target count", result.obs_counts),
        "ratio": ("Observer visibility ratio",   result.obs_ratio),
    }
    tgt_map = {
        "count":           ("Target visible observer count", result.tgt_counts),
        "ratio":           ("Target visibility ratio",        result.tgt_ratio),
        "possible_count":  ("Target possible observer count", result.tgt_possible_counts),
        "possible_ratio":  ("Target possible observer ratio", result.tgt_possible_ratio),
        "active_ratio":    ("Target active observer ratio",   result.tgt_active_ratio),
    }

    if obs_metric not in obs_map:
        raise ValueError(f"obs_metric must be one of {list(obs_map)}")
    if tgt_metric not in tgt_map:
        raise ValueError(f"tgt_metric must be one of {list(tgt_map)}")

    obs_label, obs_data = obs_map[obs_metric]
    tgt_label, tgt_data = tgt_map[tgt_metric]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    # Compute extent if georeferenced
    if dem.transform is not None:
        left, top = dem.transform * (0, 0)
        right, bottom = dem.transform * (dem.ncols, dem.nrows)
        extent = (left, right, bottom, top)
    else:
        extent = None

    # --- Left panel: observer metric ---
    im1 = ax1.imshow(obs_data, origin="upper", extent=extent)
    fig.colorbar(im1, ax=ax1, label=obs_label)
    if observer is not None:
        if dem.transform is not None:
            x, y = dem.coord(*observer)
            ax1.plot(x, y, marker="x", color="white", markersize=8)
        else:
            ax1.plot(observer[1], observer[0], marker="x", color="white", markersize=8)
    ax1.set_title(obs_label)
    ax1.set_xlabel("X")
    ax1.set_ylabel("Y")

    # --- Right panel: target metric ---
    im2 = ax2.imshow(tgt_data, origin="upper", extent=extent)
    fig.colorbar(im2, ax=ax2, label=tgt_label)
    if target_mask is not None:
        # overlay target area in red
        mask = np.ma.masked_where(~target_mask, target_mask)
        ax2.imshow(mask, origin="upper", extent=extent,
                   cmap="Reds", alpha=0.4)
    ax2.set_title(tgt_label)
    ax2.set_xlabel("X")
    ax2.set_ylabel("Y")

    plt.tight_layout()
    plt.show()