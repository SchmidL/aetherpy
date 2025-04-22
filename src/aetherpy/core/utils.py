import math
from numba import njit
import time
import functools

@njit
def euclidean_distance(r0, c0, r1, c1, res_y, res_x):
    """
    Real‑world horizontal distance between two grid cells,
    scaling row/col differences by the pixel resolution.
    """
    dy = (r1 - r0) * res_y
    dx = (c1 - c0) * res_x
    return math.hypot(dx, dy)

@njit
def compute_slope(delta_h, r0, c0, r1, c1, res_y, res_x):
    """
    Elevation rise / horizontal run between (r0,c0) and (r1,c1),
    using real‑world distance.
    """
    dist = euclidean_distance(r0, c0, r1, c1, res_y, res_x)
    if dist <= 0.0:
        return -1e9
    return delta_h / dist

def timeit(func):
    """
    Simple decorator to print the elapsed time of a function call.
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        t0 = time.perf_counter()
        result = func(*args, **kwargs)
        t1 = time.perf_counter()
        print(f"{func.__name__} took {t1 - t0:.3f} s")
        return result
    return wrapper