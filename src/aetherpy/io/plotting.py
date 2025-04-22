# aetherpy/io/plotting.py
import matplotlib.pyplot as plt
import numpy as np

def plot_viewshed(dem, vs, observer=None, figsize=(8, 6)):
    """
    Plot the DEM and overlay a viewshed mask.

    Parameters
    ----------
    dem : DEM instance
        Must have `.array`, `.nrows`, `.ncols`, and, if georeferenced,
        `.transform` & `.coord()` available.
    vs : 2D bool array
        Output of viewshed_sweep (True = visible).
    observer : tuple (row, col), optional
        Mark this point with an 'x'.
    figsize : tuple, optional
        Figure size.
    """
    fig, ax = plt.subplots(figsize=figsize)

    # Determine plotting extent
    if dem.transform is not None:
        # Map extents in real coordinates
        left, top = dem.transform * (0, 0)
        right, bottom = dem.transform * (dem.ncols, dem.nrows)
        extent = (left, right, bottom, top)
        ax.imshow(dem.array, origin='upper', extent=extent)
        # mask invisible points as NaN so they’re transparent
        mask = np.where(vs, 1.0, np.nan)
        ax.imshow(mask, origin='upper', extent=extent, alpha=0.4)
        if observer:
            x, y = dem.coord(*observer)
            ax.plot(x, y, marker='x', markersize=10)
    else:
        ax.imshow(dem.array, origin='upper')
        mask = np.where(vs, 1.0, np.nan)
        ax.imshow(mask, origin='upper', alpha=0.4)
        if observer:
            # note: (col, row) for plotting
            ax.plot(observer[1], observer[0], marker='x', markersize=10)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    plt.show()



def plot_visibility_indices(dem, result,
                            target_mask=None,
                            observer=None,
                            value="ratio",
                            figsize=(14, 6)):
    """
    Plot two side‑by‑side panels:
      Left : observer metric (ratio or count)
      Right: target metric (ratio or count), with optional mask

    Parameters
    ----------
    dem : DEM
    target_mask : 2D bool array, optional
      If given, overlays the target area on the right plot.
    observer : (row, col), optional
      Marks the observer location on the left plot.

    result : VisibilityResult
      Namedtuple with fields obs_counts, obs_ratio, tgt_counts, tgt_ratio.
    """
    import matplotlib.pyplot as plt
    import numpy as np

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    # extents
    if dem.transform:
        left, top = dem.transform * (0, 0)
        right, bottom = dem.transform * (dem.ncols, dem.nrows)
        extent = (left, right, bottom, top)
    else:
        extent = None

    # pick data based on value type
    if value == "ratio":
        obs_data = result.obs_ratio
        tgt_data = result.tgt_ratio
        obs_label = "Observer visibility ratio"
        tgt_label = "Target visibility ratio"
    elif value == "count":
        obs_data = result.obs_counts
        tgt_data = result.tgt_counts
        obs_label = "Observer visible target count"
        tgt_label = "Target visible observer count"
    else:
        raise ValueError("value must be 'ratio' or 'count'")

    # Left: observer metric
    im1 = ax1.imshow(obs_data, origin="upper", extent=extent)
    fig.colorbar(im1, ax=ax1, label=obs_label)
    if observer:
        x, y = dem.coord(*observer)
        ax1.plot(x, y, 'wx', markersize=8)
    ax1.set_title('Observer visibility index')
    ax1.set_xlabel('X'); ax1.set_ylabel('Y')

    # Right: target visibility ratio
    if tgt_data is not None:
        im2 = ax2.imshow(tgt_data, origin="upper", extent=extent)
        fig.colorbar(im2, ax=ax2, label=tgt_label)
        if target_mask is not None:
            # overlay target boundary in red
            mask = np.ma.masked_where(~target_mask, target_mask)
            ax2.imshow(mask, origin='upper', extent=extent,
                       cmap='Reds', alpha=0.4)
        ax2.set_title('Target visibility ratio')
        ax2.set_xlabel('X'); ax2.set_ylabel('Y')
    else:
        ax2.axis('off')

    plt.tight_layout()
    plt.show()
