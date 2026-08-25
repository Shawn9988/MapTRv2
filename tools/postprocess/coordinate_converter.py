import numpy as np

# WGS84椭球常数
A = 6378137.0
F = 1 / 298.257223563
E2 = 2 * F - F ** 2

def ecef2wgs84(ecef: np.ndarray) -> np.ndarray:
    """
    ECEF直角坐标转WGS84经纬高
    :param ecef: np.ndarray[N,3] 每一行 [X, Y, Z] 单位米
    :return: np.ndarray[N,3] 每一行 [lon(度), lat(度), h(米)] 经度、纬度、高程
    """
    arr = np.asarray(ecef, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr[np.newaxis, :]
    assert arr.ndim == 2 and arr.shape[1] == 3, "ecef输入必须为 N×3 二维数组"

    X, Y, Z = arr[:, 0], arr[:, 1], arr[:, 2]

    # 经度
    lon_rad = np.arctan2(Y, X)

    # 迭代求解纬度
    r = np.hypot(X, Y)
    lat_rad = np.arctan2(Z, r * (1 - E2))
    for _ in range(10):
        sin_lat = np.sin(lat_rad)
        N = A / np.sqrt(1 - E2 * sin_lat ** 2)
        lat_rad = np.arctan2(Z + E2 * N * sin_lat, r)

    # 大地高
    sin_lat = np.sin(lat_rad)
    N = A / np.sqrt(1 - E2 * sin_lat ** 2)
    h = r / np.cos(lat_rad) - N

    lat_deg = np.rad2deg(lat_rad)
    lon_deg = np.rad2deg(lon_rad)

    # 调整顺序：经度、纬度、高程
    return np.column_stack([lon_deg, lat_deg, h])


def wgs842ecef(wgs: np.ndarray) -> np.ndarray:
    """
    WGS84经纬高转ECEF直角坐标
    :param wgs: np.ndarray[N,3] 每一行 [lon(度), lat(度), h(米)] 经度、纬度、高程
    :return: np.ndarray[N,3] 每一行 [X, Y, Z] 单位米
    """
    arr = np.asarray(wgs, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr[np.newaxis, :]
    assert arr.ndim == 2 and arr.shape[1] == 3, "wgs输入必须为 N×3 二维数组 [lon, lat, h]"

    # 按新顺序拆分：lon, lat, h
    lon_deg, lat_deg, h = arr[:, 0], arr[:, 1], arr[:, 2]
    lat = np.radians(lat_deg)
    lon = np.radians(lon_deg)

    sin_lat = np.sin(lat)
    cos_lat = np.cos(lat)
    sin_lon = np.sin(lon)
    cos_lon = np.cos(lon)

    N = A / np.sqrt(1 - E2 * sin_lat ** 2)
    X = (N + h) * cos_lat * cos_lon
    Y = (N + h) * cos_lat * sin_lon
    Z = (N * (1 - E2) + h) * sin_lat

    return np.column_stack([X, Y, Z])