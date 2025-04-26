import rasterio
import numpy as np

__all__ = ["save_raster"]

def save_raster(
    out_path: str,
    array: np.ndarray,
    dem,
    dtype: str = None,
    nodata=None,
    compress: str = "lzw",
    count: int = 1,
    band: int = 1,
):
    """
    Save a 2D or 3D NumPy array to GeoTIFF, using a reference DEM for georeference.
    Automatically casts boolean arrays to uint8.

    Parameters
    ----------
    out_path : str
        Output file path.
    array : np.ndarray
        2D (H×W) or 3D (C×H×W) array.
    dem : DEM
        Provides .transform, .crs, .nrows, .ncols.
    dtype : str, optional
        Output data type (e.g. "uint8", "float32"). If None, inferred from array.
    nodata : scalar, optional
        No-data value to embed. If None and array is boolean, defaults to 0.
    compress : str
        GeoTIFF compression (e.g. "lzw", "deflate").
    count : int
        Number of bands (if array is 3D).
    band : int
        Band index for 2D arrays.
    """
    arr = np.array(array, copy=False)
    # 1) Boolean arrays → uint8
    if arr.dtype.kind == 'b':
        arr = arr.astype('uint8')
        out_dtype = 'uint8'
        if nodata is None:
            nodata = 0
    else:
        # 2) Non-boolean: use explicit dtype or the array's dtype
        out_dtype = dtype or str(arr.dtype)

    profile = {
        "driver": "GTiff",
        "height": dem.nrows,
        "width": dem.ncols,
        "count": count if arr.ndim == 3 else 1,
        "dtype": out_dtype,
        "crs": dem.crs,
        "transform": dem.transform,
        "compress": compress,
    }
    if nodata is not None:
        profile["nodata"] = nodata

    with rasterio.open(out_path, "w", **profile) as dst:
        if arr.ndim == 2:
            dst.write(arr.astype(out_dtype), band)
        elif arr.ndim == 3:
            dst.write(arr.astype(out_dtype))
        else:
            raise ValueError("Array must be 2D or 3D")
