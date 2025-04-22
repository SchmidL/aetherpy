# arguspy/io/plotting.py
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
