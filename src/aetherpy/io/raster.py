import rasterio
import numpy as np
from aetherpy.core.multiobserver import VisibilityResult
from typing import Sequence, Dict



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

def save_visibility_results(
    result: VisibilityResult,
    dem,
    out_dir: str,
    prefix: str = "visibility",
    metrics: Sequence[str] = None,
    dtype_map: Dict[str,str] = None,
    nodata_map: Dict[str,float] = None,
    compress: str = "lzw"
):
    """
    Save fields of a VisibilityResult as GeoTIFFs.

    Parameters
    ----------
    result : VisibilityResult
        The output of inverse_visibility.
    dem : DEM
        Reference DEM for georeference.
    out_dir : str
        Directory to write files into (will be created if needed).
    prefix : str
        Filename prefix for each raster.
    metrics : Sequence[str], optional
        Which fields to save. Defaults to all:
          ['obs_counts','obs_ratio',
           'tgt_counts','tgt_ratio',
           'tgt_possible_counts','tgt_possible_ratio',
           'tgt_active_ratio']
    dtype_map : dict, optional
        Map metric → output dtype string (e.g. 'uint8','float32').
        If omitted, uses:
          counts → 'uint32', ratios → 'float32'.
    nodata_map : dict, optional
        Map metric → nodata value. If omitted, defaults to 0 for counts,
        NaN for ratios.
    compress : str
        GeoTIFF compression (passed to save_raster).
    """
    os.makedirs(out_dir, exist_ok=True)

    # default field list
    all_metrics = [
        "obs_counts","obs_ratio",
        "tgt_counts","tgt_ratio",
        "tgt_possible_counts","tgt_possible_ratio",
        "tgt_active_ratio"
    ]
    if metrics is None:
        metrics = all_metrics

    # default dtypes & nodata
    if dtype_map is None:
        dtype_map = {m: ("uint32" if "counts" in m else "float32") for m in all_metrics}
    if nodata_map is None:
        nodata_map = {}
        for m in all_metrics:
            if "counts" in m:
                nodata_map[m] = 0
            else:
                nodata_map[m] = np.nan

    for m in metrics:
        if not hasattr(result, m):
            raise ValueError(f"VisibilityResult has no field '{m}'")
        arr = getattr(result, m)
        out_path = os.path.join(out_dir, f"{prefix}_{m}.tif")
        save_raster(
            out_path,
            arr,
            dem,
            dtype=dtype_map.get(m),
            nodata=nodata_map.get(m),
            compress=compress
        )