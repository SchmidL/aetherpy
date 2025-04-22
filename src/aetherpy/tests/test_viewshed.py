# tests/test_viewshed.py
import numpy as np
from arguspy.core import viewshed_sweep
from arguspy.data.loader import DEM

def test_viewshed_center_flat():
    arr = np.zeros((5,5))
    dem = DEM(arr)
    vs = viewshed_sweep(dem, (2,2))
    # on flat ground, center sees all
    assert vs.all()
