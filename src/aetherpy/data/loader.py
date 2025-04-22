import numpy as np
import rasterio

class DEM:
    """
    DEM wrapper supporting either:
      - A file path to a GeoTIFF (reads first band via rasterio)
      - A plain 2D numpy array (unit resolution, no georeference)
    Exposes:
      - array: 2D float64 elevation grid
      - transform, crs (or None)
      - res_x, res_y: pixel size in map units
      - nrows, ncols: dimensions
      - index(x,y) / coord(row,col) for georeferenced DEMs
    """
    def __init__(self, source):
        if isinstance(source, str):
            ds = rasterio.open(source)
            self.array = ds.read(1).astype(np.float64)
            self.transform = ds.transform
            self.crs = ds.crs
            ds.close()
            # pixel size: transform.a = width of a pixel; transform.e = negative of height
            self.res_x = self.transform.a
            self.res_y = abs(self.transform.e)
        elif isinstance(source, np.ndarray):
            self.array = source.astype(np.float64)
            self.transform = None
            self.crs = None
            # assume unit grid
            self.res_x = 1.0
            self.res_y = 1.0
        else:
            raise ValueError("DEM source must be a file path or a numpy array")
        self.nrows, self.ncols = self.array.shape

    def index(self, x, y):
        """
        Convert map coords (x,y) → raster indices (row, col).
        Only available if DEM was created from a file.
        """
        if self.transform is None:
            raise ValueError("No georeference: DEM built from numpy array")
        col, row = ~self.transform * (x, y)
        return int(row), int(col)

    def coord(self, row, col):
        """
        Convert raster indices (row, col) → map coords (x,y).
        Only available if DEM was created from a file.
        """
        if self.transform is None:
            raise ValueError("No georeference: DEM built from numpy array")
        x, y = self.transform * (col, row)
        return x, y

    def sample(self, row_f, col_f, method="nearest"):
        """
        Return elevation at fractional (row_f, col_f).
        method: "nearest" or "bilinear"
        """
        if method == "nearest":
            r = int(round(row_f))
            c = int(round(col_f))
            return self.array[r, c]
        elif method == "bilinear":
            # a tiny pure‑Python fallback for non‑Numba contexts
            # weights
            r0, c0 = int(row_f), int(col_f)
            dr, dc = row_f - r0, col_f - c0
            h00 = self.array[r0,   c0  ]
            h10 = self.array[r0+1, c0  ] if r0+1 < self.nrows else h00
            h01 = self.array[r0,   c0+1] if c0+1 < self.ncols else h00
            h11 = self.array[r0+1, c0+1] if (r0+1<self.nrows and c0+1<self.ncols) else h00
            return (h00 * (1-dr)*(1-dc) +
                    h10 * dr*(1-dc) +
                    h01 * (1-dr)*dc +
                    h11 * dr*dc)
        else:
            raise ValueError(f"Unknown method {method!r}")


    def rasterize_mask(self,
                       vector_path: str,
                       layer: str = None,
                       where: tuple = None,
                       all_touched: bool = False,
                       invert: bool = False) -> np.ndarray:
        """
        Rasterize a vector file (e.g. Shapefile) into a boolean mask
        matching this DEM’s grid.

        Parameters
        ----------
        vector_path : str
            Path to a vector file (Shapefile, GeoPackage, etc.).
        layer : str, optional
            Name of the layer inside a multi‑layer datasource.
        where : tuple(field_name, value), optional
            If given, only features where properties[field_name] == value
            are rasterized.
        all_touched : bool
            If True, marks all pixels touched by the geometry as True.
        invert : bool
            If True, flips the mask (i.e. background becomes True).

        Returns
        -------
        mask : 2D bool ndarray
            Same shape as `self.array`, True where the vector covers.
        """
        if self.transform is None:
            raise ValueError("DEM has no georeference; cannot rasterize vector.")

        import fiona
        from rasterio.features import rasterize

        # 1) read geometries
        with fiona.open(vector_path, layer=layer) as src:
            if where:
                field, val = where
                geoms = [
                    feat["geometry"]
                    for feat in src
                    if feat["properties"].get(field) == val
                ]
            else:
                geoms = [feat["geometry"] for feat in src]

        # 2) rasterize to DEM grid
        shapes = ((geom, 1) for geom in geoms)
        mask = rasterize(
            shapes,
            out_shape=(self.nrows, self.ncols),
            transform=self.transform,
            fill=0,
            all_touched=all_touched,
            dtype="uint8"
        ).astype(bool)

        # 3) invert if requested
        if invert:
            mask = ~mask

        return mask
