import json
import numpy as np
import os
import csv
from tools.postprocess.coordinate_converter import ecef2wgs84


def load_meta(meta_path):
    """读取meta.json，完全匹配 WriteTrajMetaJson 输出"""
    with open(meta_path, "r", encoding="utf-8") as f:
        data = json.load(f)
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


def local_frenet_to_global(local_s, d, meta):
    #MapTR样本把0-100归一化为-50~50，这里加50，还原
    s_global = local_s + meta["start_s"] + 50.0 
    return s_global, d


def frenet_to_enu(s_global: float, d: float, traj: np.ndarray):
    """
    traj: Nx4 [x, y, z, cumulative_s]
    返回 ENU [E, N, U]
    """
    n = traj.shape[0]
    s_clamp = np.clip(s_global, traj[0, 3], traj[-1, 3])
    seg_idx = 0
    for i in range(n - 1):
        s0 = traj[i, 3]
        s1 = traj[i + 1, 3]
        if s_clamp >= s0 - 1e-12 and s_clamp <= s1 + 1e-12:
            seg_idx = i
            break

    ax, ay, az, s0_seg = traj[seg_idx]
    bx, by, bz, s1_seg = traj[seg_idx + 1]
    seg_s_len = s1_seg - s0_seg
    t = (s_clamp - s0_seg) / seg_s_len if seg_s_len > 1e-12 else 0.0

    foot_x = ax + t * (bx - ax)
    foot_y = ay + t * (by - ay)
    foot_z = az + t * (bz - az)

    seg_dx = bx - ax
    seg_dy = by - ay
    seg_len_2d = np.hypot(seg_dx, seg_dy)
    if seg_len_2d < 1e-12:
        return np.array([foot_x, foot_y, foot_z])

    tx = seg_dx / seg_len_2d
    ty = seg_dy / seg_len_2d
    nx = -ty
    ny = tx
    enu_x = foot_x + d * nx
    enu_y = foot_y + d * ny
    enu_z = foot_z
    return np.array([enu_x, enu_y, enu_z])


def enu_to_ecef(enu: np.ndarray, meta):
    """
    enu: [E, N, U]
    meta 携带 ref_ecef, lon0_rad, lat0_rad
    返回原始 ECEF [X, Y, Z]
    """
    E, N, U = enu
    lon0 = meta["lon0_rad"]
    lat0 = meta["lat0_rad"]
    sin_lon = np.sin(lon0)
    cos_lon = np.cos(lon0)
    sin_lat = np.sin(lat0)
    cos_lat = np.cos(lat0)

    Rt = np.array([
        [-sin_lon, -cos_lon * sin_lat, cos_lat * cos_lon],
        [cos_lon,  -sin_lon * sin_lat, cos_lat * sin_lon],
        [0.0,       cos_lat,           sin_lat]
    ])
    delta_ecef = Rt @ np.array([E, N, U])
    ecef = meta["ref_ecef"] + delta_ecef
    return ecef


def local_frenet_to_ecef(local_s: float, d: float, z_frenet: float, meta):
    s_global, d_global = local_frenet_to_global(local_s, d, meta)
    enu_xyz = frenet_to_enu(s_global, d_global, meta["traj"])
    enu_xyz[2] = z_frenet
    ecef_xyz = enu_to_ecef(enu_xyz, meta)
    return enu_xyz, ecef_xyz


def batch_frenet_to_ecef(points: np.ndarray, meta):
    """
    points: [N,3] (local_s, d, z)
    return enu_all[N,2], ecef_all[N,3]
    """
    N = points.shape[0]
    enu_all = np.zeros((N, 2), dtype=np.float64)
    ecef_all = np.zeros((N, 3), dtype=np.float64)

    for i in range(N):
        ls, d = points[i]
        z = 0
        enu, ecef = local_frenet_to_ecef(ls, d, z, meta)
        enu_all[i] = enu[0:2]
        ecef_all[i] = ecef
    wgs_points = ecef2wgs84(ecef_all)
    return wgs_points, ecef_all


def line2d(pts):
    fmt = "{:.6f}"
    coords = ", ".join([f"{fmt.format(x)} {fmt.format(y)}" for x, y in pts])
    return f"LINESTRING ({coords})"


def str2d(pts):
    fmt = "{:.6f}"
    return ";".join([f"{fmt.format(x)},{fmt.format(y)}" for x, y in pts])


def line3d(pts):
    fmt = "{:.6f}"
    coords = ", ".join([f"{fmt.format(x)} {fmt.format(y)} {fmt.format(z)}" for x, y, z in pts])
    return f"LINESTRING Z ({coords})"


def str3d(pts):
    fmt = "{:.6f}"
    return ";".join([f"{fmt.format(x)},{fmt.format(y)},{fmt.format(z)}" for x, y, z in pts])


def save_pred_csv_xd(patch, csv_path, meta_json_path, pred_items):
    if meta_json_path is None or not os.path.exists(meta_json_path):
        return

    meta = load_meta(meta_json_path)

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "patch",
            "id",
            "type",
            "score",
            "frenet_wkt2d",
            "frenet_pts",
            "enu_wkt3d",
            "enu_pts3d",
            "ecef_wkt3d",
            "ecef_pts3d"
        ])

        for pred_id, pred_type, pred_score, frenet_pts in pred_items:
            pts_2d = frenet_pts[:, :2]
            enu_arr, ecef_arr = batch_frenet_to_ecef(frenet_pts, meta)

            row = [
                patch,
                pred_id,
                pred_type,
                f"{pred_score:.6f}",
                line2d(pts_2d),
                str2d(pts_2d),
                line3d(enu_arr),
                str3d(enu_arr),
                line3d(ecef_arr),
                str3d(ecef_arr)
            ]
            writer.writerow(row)


def merge_all_xd_csv(sample_dirs, base_file_na, out_merge_csv):
    """
    读取文件夹下所有 csv，拼接输出一个大 csv
    """
    all_rows = []
    header = None
    for sample_dir in sample_dirs:
        csv_path = os.path.join(sample_dir, base_file_na)
        if not os.path.exists(csv_path):
            continue
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            for idx, row in enumerate(reader):
                if idx == 0:
                    if header is None:
                        header = row
                    continue
                all_rows.append(row)

    with open(out_merge_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(all_rows)


def save_gt_csv_xd(patch, csv_path, meta_json_path, gt_items):
    if meta_json_path is None or not os.path.exists(meta_json_path):
        return

    meta = load_meta(meta_json_path)

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "patch",
            "id",
            "type",
            "frenet_wkt2d",
            "frenet_pts",
            "enu_wkt3d",
            "enu_pts3d",
            "ecef_wkt3d",
            "ecef_pts3d"
        ])

        for pred_id, pred_type, pred_score, frenet_pts in gt_items:
            pts_2d = frenet_pts[:, :2]
            enu_arr, ecef_arr = batch_frenet_to_ecef(frenet_pts, meta)

            row = [
                patch,
                pred_id,
                pred_type,
                line2d(pts_2d),
                str2d(pts_2d),
                line3d(enu_arr),
                str3d(enu_arr),
                line3d(ecef_arr),
                str3d(ecef_arr)
            ]
            writer.writerow(row)
