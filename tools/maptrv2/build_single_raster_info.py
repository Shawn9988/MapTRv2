import argparse
import os
import os.path as osp
import shutil

import mmcv
import numpy as np
from PIL import Image


def parse_args():
    parser = argparse.ArgumentParser(
        description='Build a one-sample raster infos pkl for MapTR inference.')
    parser.add_argument(
        '--image',
        required=True,
        help='input raster png/jpg path')
    parser.add_argument(
        '--out-dir',
        default='work_dirs/vendor_single_raster',
        help='output directory')
    parser.add_argument(
        '--pc-range',
        type=float,
        nargs=4,
        default=[-30.0, -40.0, 30.0, 40.0],
        help='x_min y_min x_max y_max in local coordinates')
    parser.add_argument(
        '--raw-xy',
        action='store_true',
        help='set if image horizontal axis is local x and vertical axis is local y')
    parser.add_argument(
        '--sample-name',
        default='vendor_single',
        help='sample id/name used in output infos')
    parser.add_argument(
        '--map-location',
        default=None,
        help='optional map location name for WGS84 conversion')
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    samples_dir = osp.join(args.out_dir, 'samples')
    labels_dir = osp.join(args.out_dir, 'labels')
    os.makedirs(samples_dir, exist_ok=True)
    os.makedirs(labels_dir, exist_ok=True)

    image = Image.open(args.image).convert('L')
    image_size = list(image.size)

    sample_path = osp.join(samples_dir, f'{args.sample_name}.png')
    label_path = osp.join(labels_dir, f'{args.sample_name}.png')
    shutil.copyfile(args.image, sample_path)
    Image.new('L', tuple(image_size), 0).save(label_path)

    pc_range_4d = list(args.pc_range)
    pc_range_6d = [
        pc_range_4d[0], pc_range_4d[1], -10.0,
        pc_range_4d[2], pc_range_4d[3], 10.0]
    lidar2global = np.eye(4, dtype=np.float32)
    empty_annotation = {
        'divider': [],
        'ped_crossing': [],
        'boundary': [],
        'centerline': [],
    }

    info = dict(
        sample_name=args.sample_name,
        token=args.sample_name,
        scene_token='vendor_single_scene',
        timestamp=0,
        bev_img_path=sample_path,
        boundary_label_path=label_path,
        check_img_path=sample_path,
        source_npz_path=None,
        pc_range=pc_range_6d,
        image_size=image_size,
        raw_xy=args.raw_xy,
        patch_mode='single',
        patch_order=0,
        anchor_index=0,
        stride_m=0.0,
        map_location=args.map_location,
        lidar2global=lidar2global.astype(float).tolist(),
        ego2global_translation=[0.0, 0.0, 0.0],
        ego2global_rotation=[1.0, 0.0, 0.0, 0.0],
        lidar2ego_translation=[0.0, 0.0, 0.0],
        lidar2ego_rotation=[1.0, 0.0, 0.0, 0.0],
        annotation=empty_annotation,
        global_annotation=empty_annotation,
        raster_annotation=empty_annotation,
        global_raster_annotation=empty_annotation)

    raster_data = dict(
        infos=[info],
        metadata=dict(
            source_image=args.image,
            pc_range=pc_range_4d,
            image_size=image_size,
            raw_xy=args.raw_xy,
            patch_mode='single',
            map_classes=['divider', 'ped_crossing', 'boundary', 'centerline']))
    mmcv.dump(raster_data, osp.join(args.out_dir, 'maptr_raster_infos.pkl'))
    mmcv.dump(raster_data, osp.join(args.out_dir, 'maptr_raster_infos.json'))
    print(f'Wrote single raster infos to {args.out_dir}')


if __name__ == '__main__':
    main()
