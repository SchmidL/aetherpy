# tests/test_los.py
import numpy as np
from arguspy.core import is_visible
from arguspy.data.loader import DEM

def test_simple_hill():
    dem = DEM(np.array([
        [0, 0, 0],
        [0, 5, 0],
        [0, 0, 0],
    ]))
    # (0,0) should not see (2,2) because of the hill at (1,1)
    assert not is_visible(dem, (0,0), (2,2))
    # the same hill also blocks the other diagonal
    assert not is_visible(dem, (0,2), (2,0))