import json
import numpy as np

def load_meta(meta_path):
    """读取meta.json，完全匹配WriteTrajMetaJson输出"""
    with open(meta_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    # 轨迹折线 Nx4: [x, y, z, cumulative_s]
    traj = np.array(data["traj_polyline_enu"], dtype=np.float64)
    ori = data["enu_origin"]
    return {
        "start_s": data["window_s_range"]["start_s"],
        "end_s": data["window_s_range"]["end_s"],
        "ref_ecef": np.array([ori["ecef_x0"], ori["ecef_y0"], ori["ecef_z0"]]),
        "lon0_rad": ori["lon0_rad"],
        "lat0_rad": ori["lat0_rad"],
        "traj": traj
    }

#逆函数 1：局部 Frenet → 全局 Frenet（反向 CollectGtBoundaries 里s -= start_s）
def local_frenet_to_global(local_s, d, meta):
    s_global = local_s + meta["start_s"]
    return s_global, d

#逆函数 2：全局 Frenet (s_global, d) → ENU (x,y,z)（1:1 复刻 C++ frenetToENU）
def frenet_to_enu(s_global: float, d: float, traj: np.ndarray):
    """
    traj: Nx4 [x, y, z, cumulative_s]
    返回ENU [E, N, U]
    """
    n = traj.shape[0]
    s_clamp = np.clip(s_global, traj[0, 3], traj[-1, 3])
    seg_idx = 0
    # 查找所在线段
    for i in range(n - 1):
        s0 = traj[i, 3]
        s1 = traj[i + 1, 3]
        if s_clamp >= s0 - 1e-12 and s_clamp <= s1 + 1e-12:
            seg_idx = i
            break
    # 线段两点
    ax, ay, az, s0_seg = traj[seg_idx]
    bx, by, bz, s1_seg = traj[seg_idx + 1]
    seg_s_len = s1_seg - s0_seg
    t = (s_clamp - s0_seg) / seg_s_len if seg_s_len > 1e-12 else 0.0
    # 中心线垂足
    foot_x = ax + t * (bx - ax)
    foot_y = ay + t * (by - ay)
    foot_z = az + t * (bz - az)
    # 线段2D切向单位向量
    seg_dx = bx - ax
    seg_dy = by - ay
    seg_len_2d = np.hypot(seg_dx, seg_dy)
    if seg_len_2d < 1e-12:
        return np.array([foot_x, foot_y, foot_z])
    tx = seg_dx / seg_len_2d
    ty = seg_dy / seg_len_2d
    # 法向 (-ty, tx) 和C++ d计算公式符号严格一致
    nx = -ty
    ny = tx
    enu_x = foot_x + d * nx
    enu_y = foot_y + d * ny
    enu_z = foot_z
    return np.array([enu_x, enu_y, enu_z])

#逆函数 3：ENU → ECEF（完全反向 C++ ecefToENU）
def enu_to_ecef(enu: np.ndarray, meta):
    """
    enu: [E, N, U]
    meta携带 ref_ecef, lon0_rad, lat0_rad
    返回原始ECEF [X, Y, Z]
    """
    E, N, U = enu
    lon0 = meta["lon0_rad"]
    lat0 = meta["lat0_rad"]
    sin_lon = np.sin(lon0)
    cos_lon = np.cos(lon0)
    sin_lat = np.sin(lat0)
    cos_lat = np.cos(lat0)
    # R^T 旋转矩阵，和C++ ecefToENU互逆
    Rt = np.array([
        [-sin_lon, -cos_lon * sin_lat, cos_lat * cos_lon],
        [cos_lon,  -sin_lon * sin_lat, cos_lat * sin_lon],
        [0.0,       cos_lat,           sin_lat]
    ])
    delta_ecef = Rt @ np.array([E, N, U])
    ecef = meta["ref_ecef"] + delta_ecef
    return ecef

def local_frenet_to_ecef(local_s: float, d: float, z_frenet: float, meta):
    # 步骤1：局部s -> 全局s
    s_global, d_global = local_frenet_to_global(local_s, d, meta)
    # 步骤2：Frenet(s_global, d) -> ENU
    enu_xyz = frenet_to_enu(s_global, d_global, meta["traj"])
    # z可替换为frenet里的z，这里复用插值foot_z，按需覆盖
    enu_xyz[2] = z_frenet
    # 步骤3：ENU -> 原始ECEF大地坐标
    ecef_xyz = enu_to_ecef(enu_xyz, meta)
    return enu_xyz, ecef_xyz

def batch_frenet_to_ecef(points: np.ndarray, meta):
    """
    points: [N,3] (local_s, d, z)
    return enu_all[N,3], ecef_all[N,3]
    """
    N = points.shape[0]
    enu_all = np.zeros_like(points)
    ecef_all = np.zeros_like(points)
    for i in range(N):
        ls, d, z = points[i]
        enu, ecef = local_frenet_to_ecef(ls, d, z, meta)
        enu_all[i] = enu
        ecef_all[i] = ecef
    return enu_all, ecef