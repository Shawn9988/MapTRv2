import csv
import numpy as np
import os.path as osp

# ========== nuScenes 专用CSV工具（原生逻辑，不混入自有数据集转换） ==========
NUSCENES_MAP_ORIGINS = {
    'boston-seaport': (42.336849169438615, -71.05785369873047),
    'singapore-hollandvillage': (1.2993652317780957, 103.78217697143555),
    'singapore-onenorth': (1.2882100868743724, 103.78475189208984),
    'singapore-queenstown': (1.2782562240223188, 103.76741409301758),
}

def linestring_wkt(points, precision=4):
    fmt = f'{{:.{precision}f}}'
    coords = ', '.join([f'{fmt.format(float(x))} {fmt.format(float(y))}' for x, y in points[:, :2]])
    return f'LINESTRING ({coords})'

def points_literal(points, precision=4):
    fmt = f'{{:.{precision}f}}'
    return ';'.join([f'{fmt.format(float(x))},{fmt.format(float(y))}' for x, y in points[:, :2]])

def global_to_wgs84_points(points, map_location):
    points = np.asarray(points, dtype=np.float32)
    if points.ndim != 2:
        points = points.reshape(-1, 2)
    if map_location not in NUSCENES_MAP_ORIGINS:
        return None
    lat0, lon0 = NUSCENES_MAP_ORIGINS[map_location]
    meters_per_deg_lat = 111320.0
    meters_per_deg_lon = meters_per_deg_lat * np.cos(np.deg2rad(lat0))
    lon = lon0 + points[:, 0] / meters_per_deg_lon
    lat = lat0 + points[:, 1] / meters_per_deg_lat
    return np.stack([lon, lat], axis=1)

def as_numpy_points(points):
    if hasattr(points, 'cpu'):
        points = points.cpu().numpy()
    points = np.asarray(points, dtype=np.float32)
    if points.ndim != 2:
        points = points.reshape(-1, 2)
    return points[:, :2]

def local_to_global_points(points, lidar2global):
    points = as_numpy_points(points)
    lidar2global = np.asarray(lidar2global, dtype=np.float32)
    if lidar2global.shape != (4, 4):
        return points
    points_3d = np.zeros((points.shape[0], 3), dtype=np.float32)
    points_3d[:, :2] = points[:, :2]
    points_h = np.concatenate([points_3d, np.ones((points_3d.shape[0], 1), dtype=np.float32)], axis=1)
    global_points = points_h @ lidar2global.T
    return global_points[:, :2]

def save_pred_csv(csv_path, pred_items, lidar2global=None, map_location=None):
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'id', 'type', 'score',
            'wkt_local', 'points_local',
            'wkt_global', 'points_global',
            'wkt_wgs84', 'points_wgs84'])
        for pred_id, pred_type, pred_score, points in pred_items:
            global_points = local_to_global_points(points, lidar2global) if lidar2global is not None else points
            wgs84_points = global_to_wgs84_points(global_points, map_location)
            writer.writerow([
                pred_id,
                pred_type,
                f'{float(pred_score):.6f}',
                linestring_wkt(points),
                points_literal(points),
                linestring_wkt(global_points),
                points_literal(global_points),
                linestring_wkt(wgs84_points, 7) if wgs84_points is not None else '',
                points_literal(wgs84_points, 7) if wgs84_points is not None else '',
            ])

def save_gt_global_csv(csv_path, gt_rows, map_location=None):
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'id', 'type',
            'wkt_local', 'points_local',
            'wkt_global', 'points_global',
            'wkt_wgs84', 'points_wgs84'])
        for row_id, (map_type, line_id, local_points, global_points) in enumerate(gt_rows):
            wgs84_points = global_to_wgs84_points(global_points, map_location)
            writer.writerow([
                row_id,
                map_type,
                linestring_wkt(local_points),
                points_literal(local_points),
                linestring_wkt(global_points),
                points_literal(global_points),
                linestring_wkt(wgs84_points, 7) if wgs84_points is not None else '',
                points_literal(wgs84_points, 7) if wgs84_points is not None else '',
            ])

def save_all_pred_csv(all_pred_csv_path, all_pred_items):
    with open(all_pred_csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'sample_id', 'map_location', 'id', 'type', 'score',
            'wkt_local', 'points_local',
            'wkt_global', 'points_global',
            'wkt_wgs84', 'points_wgs84'])
        for sample_id, map_location, pred_id, pred_type, pred_score, points, lidar2global in all_pred_items:
            global_points = local_to_global_points(points, lidar2global)
            wgs84_points = global_to_wgs84_points(global_points, map_location)
            writer.writerow([
                sample_id,
                map_location,
                pred_id,
                pred_type,
                f'{float(pred_score):.6f}',
                linestring_wkt(points),
                points_literal(points),
                linestring_wkt(global_points),
                points_literal(global_points),
                linestring_wkt(wgs84_points, 7) if wgs84_points is not None else '',
                points_literal(wgs84_points, 7) if wgs84_points is not None else '',
            ])

def save_all_gt_csv(all_gt_csv_path, all_gt_items):
    with open(all_gt_csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'sample_id', 'map_location', 'id', 'type',
            'wkt_local', 'points_local',
            'wkt_global', 'points_global',
            'wkt_wgs84', 'points_wgs84'])
        for row_id, (sample_id, map_location, map_type, line_id, local_points, global_points) in enumerate(all_gt_items):
            wgs84_points = global_to_wgs84_points(global_points, map_location)
            writer.writerow([
                sample_id,
                map_location,
                row_id,
                map_type,
                linestring_wkt(local_points),
                points_literal(local_points),
                linestring_wkt(global_points),
                points_literal(global_points),
                linestring_wkt(wgs84_points, 7) if wgs84_points is not None else '',
                points_literal(wgs84_points, 7) if wgs84_points is not None else '',
            ])