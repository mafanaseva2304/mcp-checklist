import numpy as np
from collections import Counter


def mad(a, axis=0):
    med = np.nanmedian(a, axis=axis, keepdims=True)
    return np.nanmedian(np.abs(a - med), axis=axis)

def rolling_mad(arr, win):
    """
    arr shape: (n,)            – один тег
    return: shape (n,)      – MAD в каждом okне длиной win
    min_periods = 1: для первых k<win точек
    """
    n, out = arr.shape[0], np.empty(arr.shape[0], float)
    for i in range(n):
        lo = max(0, i - win + 1)
        out[i] = mad(arr[lo:i + 1])
    return out

def hysteresis_mode75(series: np.ndarray, buf: int)-> np.ndarray:
    """
    series: 1d float array (np.nan, 1, 2, 3)
    buf   : длина окна (int)
    """
    n = series.size
    res, buf_arr = np.full(n, np.nan), np.full(buf, np.nan)
    for i, v in enumerate(series):
        buf_arr[i % buf] = v
        valid = buf_arr[~np.isnan(buf_arr)]
        if valid.size:
            top, cnt = Counter(valid).most_common(1)[0]
            if cnt / buf_arr.size >= 0.75:
                res[i] = top
    return res

def mad_z(arr, median, mad_val):
    if mad_val == 0:
        return np.zeros_like(arr)
    return 0.6745 * (arr - median) / mad_val