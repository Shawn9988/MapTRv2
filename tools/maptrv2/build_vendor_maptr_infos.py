import argparse
import json
import os
import os.path as osp

import mmcv
import numpy as np
from PIL import Image


def parse_args():
    parser = argparse.ArgumentParser(
        description='Convert vendor MapTR sample folders to maptr_raster_infos.pkl.')
    parser.add_argument(
        '--sample-root',
        required=True,
        help='directory containing *_sample_xxxxxx folders')
    parser.add_argument(
        '--out-dir',
        required=True,
        help='output directory for maptr_raster_infos.pkl/json')
    parser.add_argument(
        '--pc-range',
        type=float,
        nargs=4,
        default=[0.0, -40.0, 100.0, 40.0],
        help='s_min d_min s_max d_max in sample local Frenet coordinates')
    parser.add_argument(
        '--max-samples',
        type=int,
        default=None)
    return parser.parse_args()


def find_sample_dirs(sample_root):
    sample_dirs = []
    for root, dirs, files in os.walk(sample_root):
        if 'sample.json' in files and 'raster.png' in files:
            sample_dirs.append(root)
            dirs[:] = []
    return sorted(sample_dirs)


def load_boundaries(sample_json):
    with open(sample_json, 'r', encoding='utf-8') as f:
        data = json.load(f)
    boundaries = []
    for item in data.get('gt_boundaries', []):
        points = np.asarray(item.get('points_xyz_m', []), dtype=np.float32)
        if points.ndim != 2 or points.shape[0] < 2:
            continue
        boundaries.append(points[:, :2].astype(float).tolist())
    return boundaries, data


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    sample_dirs = find_sample_dirs(args.sample_root)
    if args.max_samples is not None:
        sample_dirs = sample_dirs[:args.max_samples]

    pc_range_4d = list(args.pc_range)
    pc_range_6d = [
        pc_range_4d[0], pc_range_4d[1], -10.0,
        pc_range_4d[2], pc_range_4d[3], 10.0]
    lidar2global = np.eye(4, dtype=np.float32)

    infos = []
    for sample_index, sample_dir in enumerate(sample_dirs):
        sample_json = osp.join(sample_dir, 'sample.json')
        raster_path = osp.join(sample_dir, 'raster.png')
        debug_path = osp.join(sample_dir, 'raster_with_boundary.png')
        boundaries, raw_json = load_boundaries(sample_json)
        image_size = list(Image.open(raster_path).size)

        sample_name = osp.basename(sample_dir)
        annotation = {
            'divider': [],
            'ped_crossing': [],
            'boundary': boundaries,
            'centerline': [],
        }
        info = dict(
            sample_name=sample_name,
            token=raw_json.get('token', sample_name),
            scene_token=raw_json.get('scene_token', 'vendor_scene'),
            timestamp=raw_json.get('timestamp', sample_index),
            bev_img_path=raster_path,
            boundary_label_path=None,
            check_img_path=debug_path if osp.exists(debug_path) else raster_path,
            source_npz_path=None,
            pc_range=pc_range_6d,
            image_size=image_size,
            raw_xy=True,
            patch_mode=raw_json.get('patch_mode', 'vendor_strip'),
            patch_order=raw_json.get('patch_order', sample_index),
            anchor_index=raw_json.get('anchor_index', sample_index),
            stride_m=raw_json.get('stride_m', 0.0),
            map_location=raw_json.get('map_location', None),
            lidar2global=raw_json.get(
                'local_to_global',
                lidar2global.astype(float).tolist()),
            ego2global_translation=raw_json.get(
                'ego2global_translation', [0.0, 0.0, 0.0]),
            ego2global_rotation=raw_json.get(
                'ego2global_rotation', [1.0, 0.0, 0.0, 0.0]),
            lidar2ego_translation=[0.0, 0.0, 0.0],
            lidar2ego_rotation=[1.0, 0.0, 0.0, 0.0],
            annotation=annotation,
            global_annotation=annotation,
            raster_annotation=annotation,
            global_raster_annotation=annotation)
        infos.append(info)

    raster_data = dict(
        infos=infos,
        metadata=dict(
            source_sample_root=args.sample_root,
            pc_range=pc_range_4d,
            image_size=None,
            raw_xy=True,
            patch_mode='vendor',
            map_classes=['divider', 'ped_crossing', 'boundary', 'centerline']))
    mmcv.dump(raster_data, osp.join(args.out_dir, 'maptr_raster_infos.pkl'))

    json_infos = []
    for item in infos:
        json_item = item.copy()
        json_infos.append(json_item)
    mmcv.dump(
        dict(infos=json_infos, metadata=raster_data['metadata']),
        osp.join(args.out_dir, 'maptr_raster_infos.json'))
    print(f'Exported {len(infos)} vendor samples to {args.out_dir}')


if __name__ == '__main__':
    main()
