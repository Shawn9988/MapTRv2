import argparse
import csv
from fileinput import filename
import mmcv
import os
import shutil
import torch
import warnings
from mmcv import Config, DictAction
from mmcv.cnn import fuse_conv_bn
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner import (get_dist_info, init_dist, load_checkpoint,
                         wrap_fp16_model)
from mmdet3d.utils import collect_env, get_root_logger
from mmdet3d.apis import single_gpu_test
from mmdet3d.datasets import build_dataset
from projects.mmdet3d_plugin.datasets.builder import build_dataloader
from mmdet3d.models import build_model
from mmdet.apis import set_random_seed
from projects.mmdet3d_plugin.bevformer.apis.test import custom_multi_gpu_test
from mmdet.datasets import replace_ImageToTensor
import time
import os.path as osp
import numpy as np
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
from matplotlib import transforms
from matplotlib.patches import Rectangle
import cv2
from tools.postprocess.xd_to_csv import save_pred_csv_xd, save_gt_csv_xd, merge_all_xd_csv

CAMS = ['CAM_FRONT_LEFT','CAM_FRONT','CAM_FRONT_RIGHT',
             'CAM_BACK_LEFT','CAM_BACK','CAM_BACK_RIGHT',]
# we choose these samples not because it is easy but because it is hard
CANDIDATE=['n008-2018-08-01-15-16-36-0400_1533151184047036',
           'n008-2018-08-01-15-16-36-0400_1533151200646853',
           'n008-2018-08-01-15-16-36-0400_1533151274047332',
           'n008-2018-08-01-15-16-36-0400_1533151369947807',
           'n008-2018-08-01-15-16-36-0400_1533151581047647',
           'n008-2018-08-01-15-16-36-0400_1533151585447531',
           'n008-2018-08-01-15-16-36-0400_1533151741547700',
           'n008-2018-08-01-15-16-36-0400_1533151854947676',
           'n008-2018-08-22-15-53-49-0400_1534968048946931',
           'n008-2018-08-22-15-53-49-0400_1534968255947662',
           'n008-2018-08-01-15-16-36-0400_1533151616447606',
           'n015-2018-07-18-11-41-49+0800_1531885617949602',
           'n008-2018-08-28-16-43-51-0400_1535489136547616',
           'n008-2018-08-28-16-43-51-0400_1535489145446939',
           'n008-2018-08-28-16-43-51-0400_1535489152948944',
           'n008-2018-08-28-16-43-51-0400_1535489299547057',
           'n008-2018-08-28-16-43-51-0400_1535489317946828',
           'n008-2018-09-18-15-12-01-0400_1537298038950431',
           'n008-2018-09-18-15-12-01-0400_1537298047650680',
           'n008-2018-09-18-15-12-01-0400_1537298056450495',
           'n008-2018-09-18-15-12-01-0400_1537298074700410',
           'n008-2018-09-18-15-12-01-0400_1537298088148941',
           'n008-2018-09-18-15-12-01-0400_1537298101700395',
           'n015-2018-11-21-19-21-35+0800_1542799330198603',
           'n015-2018-11-21-19-21-35+0800_1542799345696426',
           'n015-2018-11-21-19-21-35+0800_1542799353697765',
           'n015-2018-11-21-19-21-35+0800_1542799525447813',
           'n015-2018-11-21-19-21-35+0800_1542799676697935',
           'n015-2018-11-21-19-21-35+0800_1542799758948001',
           ]

def perspective(cam_coords, proj_mat):
    pix_coords = proj_mat @ cam_coords
    valid_idx = pix_coords[2, :] > 0
    pix_coords = pix_coords[:, valid_idx]
    pix_coords = pix_coords[:2, :] / (pix_coords[2, :] + 1e-7)
    pix_coords = pix_coords.transpose(1, 0)
    return pix_coords

def parse_args():
    parser = argparse.ArgumentParser(description='vis hdmaptr map gt label')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument('--score-thresh', default=0.4, type=float, help='samples to visualize')
    parser.add_argument(
        '--show-dir', help='directory where visualizations will be saved')
    parser.add_argument('--show-cam', action='store_true', help='show camera pic')
    parser.add_argument(
        '--max-samples',
        default=-1,
        type=int,
        help='maximum number of samples to visualize; -1 means all samples')
    parser.add_argument(
        '--gt-format',
        type=str,
        nargs='+',
        default=['fixed_num_pts',],
        help='vis format, default should be "points",'
        'support ["se_pts","bbox","fixed_num_pts","polyline_pts"]')
    parser.add_argument(
        '--meta-root',
        type=str,
        # default=None,
        default="/data/huinian/data/MapTR/0710/meta",
        help='自有数据集meta.json顶层目录，样本文件夹一一对应'
    )
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    cfg = Config.fromfile(args.config)

    # import modules from plguin/xx, registry will be updated
    if hasattr(cfg, 'plugin'):
        if cfg.plugin:
            import importlib
            if hasattr(cfg, 'plugin_dir'):
                plugin_dir = cfg.plugin_dir
                _module_dir = os.path.dirname(plugin_dir)
                _module_dir = _module_dir.split('/')
                _module_path = _module_dir[0]

                for m in _module_dir[1:]:
                    _module_path = _module_path + '.' + m
                print(_module_path)
                plg_lib = importlib.import_module(_module_path)
            else:
                # import dir is the dirpath for the config file
                _module_dir = os.path.dirname(args.config)
                _module_dir = _module_dir.split('/')
                _module_path = _module_dir[0]
                for m in _module_dir[1:]:
                    _module_path = _module_path + '.' + m
                print(_module_path)
                plg_lib = importlib.import_module(_module_path)

    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True

    cfg.model.pretrained = None
    # in case the test dataset is concatenated
    samples_per_gpu = 1
    if isinstance(cfg.data.test, dict):
        cfg.data.test.test_mode = True
        samples_per_gpu = cfg.data.test.pop('samples_per_gpu', 1)
        if samples_per_gpu > 1:
            # Replace 'ImageToTensor' to 'DefaultFormatBundle'
            cfg.data.test.pipeline = replace_ImageToTensor(
                cfg.data.test.pipeline)
    elif isinstance(cfg.data.test, list):
        for ds_cfg in cfg.data.test:
            ds_cfg.test_mode = True
        samples_per_gpu = max(
            [ds_cfg.pop('samples_per_gpu', 1) for ds_cfg in cfg.data.test])
        if samples_per_gpu > 1:
            for ds_cfg in cfg.data.test:
                ds_cfg.pipeline = replace_ImageToTensor(ds_cfg.pipeline)

    if args.show_dir is None:
        args.show_dir = osp.join('./work_dirs', 
                                osp.splitext(osp.basename(args.config))[0],
                                'vis_pred')
    # create vis_label dir
    if osp.exists(args.show_dir):
        show_dir_abs = osp.abspath(args.show_dir)
        cwd_abs = osp.abspath(os.getcwd())
        if show_dir_abs in (osp.abspath(os.sep), cwd_abs):
            raise ValueError(f'Refuse to clear unsafe show-dir: {show_dir_abs}')
        shutil.rmtree(show_dir_abs)
    mmcv.mkdir_or_exist(osp.abspath(args.show_dir))
    cfg.dump(osp.join(args.show_dir, osp.basename(args.config)))
    logger = get_root_logger()
    logger.info(f'DONE create vis_pred dir: {args.show_dir}')


    dataset = build_dataset(cfg.data.test)
    dataset.is_vis_on_test = True #TODO, this is a hack
    data_loader = build_dataloader(
        dataset,
        samples_per_gpu=samples_per_gpu,
        # workers_per_gpu=cfg.data.workers_per_gpu,
        workers_per_gpu=0,
        dist=False,
        shuffle=False,
        nonshuffler_sampler=cfg.data.nonshuffler_sampler,
    )
    logger.info('Done build test data set')

    # build the model and load checkpoint
    # import pdb;pdb.set_trace()
    cfg.model.train_cfg = None
    # cfg.model.pts_bbox_head.bbox_coder.max_num=15 # TODO this is a hack
    model = build_model(cfg.model, test_cfg=cfg.get('test_cfg'))
    fp16_cfg = cfg.get('fp16', None)
    if fp16_cfg is not None:
        wrap_fp16_model(model)
    logger.info('loading check point')
    checkpoint = load_checkpoint(model, args.checkpoint, map_location='cpu')
    if 'CLASSES' in checkpoint.get('meta', {}):
        model.CLASSES = checkpoint['meta']['CLASSES']
    else:
        model.CLASSES = dataset.CLASSES
    # palette for visualization in segmentation tasks
    if 'PALETTE' in checkpoint.get('meta', {}):
        model.PALETTE = checkpoint['meta']['PALETTE']
    elif hasattr(dataset, 'PALETTE'):
        # segmentation dataset has `PALETTE` attribute
        model.PALETTE = dataset.PALETTE
    logger.info('DONE load check point')
    model = MMDataParallel(model, device_ids=[0])
    model.eval()

    img_norm_cfg = cfg.img_norm_cfg

    # get denormalized param
    mean = np.array(img_norm_cfg['mean'],dtype=np.float32)
    std = np.array(img_norm_cfg['std'],dtype=np.float32)
    to_bgr = img_norm_cfg['to_rgb']

    # get pc_range
    pc_range = cfg.point_cloud_range

    # get car icon
    car_img = Image.open('./figs/lidar_car.png')
    if max(car_img.size) > 256:
        car_img = None

    # get color map: divider->r, ped->b, boundary->g
    colors_plt = ['orange', 'b', 'g']


    logger.info('BEGIN vis test dataset samples gt label & pred')



    bbox_results = []
    mask_results = []
    dataset = data_loader.dataset
    have_mask = False
    # prog_bar = mmcv.ProgressBar(len(CANDIDATE))
    max_samples = len(dataset) if args.max_samples < 0 else min(args.max_samples, len(dataset))
    prog_bar = mmcv.ProgressBar(max_samples)
    # import pdb;pdb.set_trace()
    vis_count = 0

    def unwrap_dc(obj):
        return obj.data if hasattr(obj, 'data') else obj

    def first_batch(obj):
        obj = unwrap_dc(obj)
        if isinstance(obj, list) and len(obj) > 0 and hasattr(obj[0], 'data'):
            obj = unwrap_dc(obj[0])
        if isinstance(obj, list) and len(obj) > 0 and isinstance(obj[0], list):
            return obj[0]
        return obj

    def meta_list(obj):
        obj = first_batch(obj)
        if isinstance(obj, dict):
            return [obj]
        return obj

    def batch_list(obj):
        obj = first_batch(obj)
        if not isinstance(obj, (list, tuple)):
            return [obj]
        return obj

    def label_name(label):
        label = int(label)
        map_classes = getattr(dataset, 'MAPCLASSES', None)
        if map_classes is not None and label < len(map_classes):
            return str(map_classes[label])
        classes = getattr(dataset, 'CLASSES', None)
        if classes is not None and label < len(classes):
            return str(classes[label])
        return str(label)

    def linestring_wkt(points, precision=4):
        fmt = f'{{:.{precision}f}}'
        coords = ', '.join([f'{fmt.format(float(x))} {fmt.format(float(y))}'
                            for x, y in points[:, :2]])
        return f'LINESTRING ({coords})'

    def points_literal(points, precision=4):
        fmt = f'{{:.{precision}f}}'
        return ';'.join([f'{fmt.format(float(x))},{fmt.format(float(y))}'
                         for x, y in points[:, :2]])

    # nuScenes map/global coordinates are local ENU-like meter coordinates.
    # These reference coordinates are the WGS84 lat/lon anchors used by the
    # nuScenes devkit export utilities. WKT uses x/y order, so WGS84 WKT is
    # written as lon lat.
    NUSCENES_MAP_ORIGINS = {
        'boston-seaport': (42.336849169438615, -71.05785369873047),
        'singapore-hollandvillage': (1.2993652317780957, 103.78217697143555),
        'singapore-onenorth': (1.2882100868743724, 103.78475189208984),
        'singapore-queenstown': (1.2782562240223188, 103.76741409301758),
    }

    def global_to_wgs84_points(points, map_location):
        points = as_numpy_points(points)
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
        points_h = np.concatenate(
            [points_3d, np.ones((points_3d.shape[0], 1), dtype=np.float32)],
            axis=1)
        global_points = points_h @ lidar2global.T
        return global_points[:, :2]

    def annotation_lines(annotation):
        rows = []
        if not isinstance(annotation, dict):
            return rows
        for map_type in sorted(annotation.keys()):
            lines = annotation.get(map_type, [])
            for line_id, points in enumerate(lines):
                points = as_numpy_points(points)
                if points.shape[0] < 2:
                    continue
                rows.append((line_id, map_type, points))
        return rows

    def build_gt_rows(local_annotation, global_annotation, lidar2global):
        global_by_key = {}
        for line_id, map_type, points in annotation_lines(global_annotation):
            global_by_key[(map_type, line_id)] = points

        rows = []
        for line_id, map_type, local_points in annotation_lines(local_annotation):
            global_points = global_by_key.get((map_type, line_id))
            if global_points is None:
                global_points = local_to_global_points(local_points, lidar2global)
            rows.append((line_id, map_type, local_points, global_points))
        return rows

    def xy_to_pixel(points, image_size, pc_range, raw_xy=False):
        width, height = image_size
        x_min, y_min, _, x_max, y_max, _ = pc_range
        if raw_xy:
            disp_points = points[:, [0, 1]]
            disp_x_min, disp_x_max = x_min, x_max
            disp_y_min, disp_y_max = y_min, y_max
        else:
            # Match tools/maptrv2/export_nusc_radar_map_samples.py:
            # exported raster images use display coordinates (y, x) by default.
            disp_points = points[:, [1, 0]]
            disp_x_min, disp_x_max = y_min, y_max
            disp_y_min, disp_y_max = x_min, x_max
        xs = (disp_points[:, 0] - disp_x_min) / (disp_x_max - disp_x_min)
        ys = (disp_y_max - disp_points[:, 1]) / (disp_y_max - disp_y_min)
        pixels = np.stack([xs * (width - 1), ys * (height - 1)], axis=1)
        return np.round(pixels).astype(np.int32)

    def draw_lines_on_image(draw, lines, image_size, pc_range, raw_xy,
                            fill, width=1, point_radius=0):
        for points in lines:
            if points.shape[0] < 2:
                continue
            pixels = xy_to_pixel(points, image_size, pc_range, raw_xy=raw_xy)
            draw.line([tuple(point) for point in pixels], fill=fill, width=width)
            if point_radius > 0:
                for x, y in pixels:
                    draw.ellipse(
                        (x - point_radius, y - point_radius,
                         x + point_radius, y + point_radius),
                        fill=fill)

    def gt_lines_from_bboxes(gt_bbox):
        gt_lines = []
        if hasattr(gt_bbox, 'fixed_num_sampled_points'):
            for points in gt_bbox.fixed_num_sampled_points:
                if hasattr(points, 'cpu'):
                    points = points.cpu().numpy()
                else:
                    points = np.asarray(points)
                gt_lines.append(points[:, :2])
        elif hasattr(gt_bbox, 'instance_list'):
            for instance in gt_bbox.instance_list:
                gt_lines.append(np.asarray(list(instance.coords))[:, :2])
        return gt_lines

    def draw_pred_on_raster(raster_path, out_path, pred_items, pc_range,
                            raw_xy=False, gt_lines=None, trajectory=None):
        if raster_path is None or not osp.exists(raster_path):
            return
        img = Image.open(raster_path).convert('RGB')
        draw = ImageDraw.Draw(img)
        if trajectory is not None and len(trajectory) >= 2:
            draw_lines_on_image(
                draw, [trajectory], img.size, pc_range, raw_xy,
                fill=(0, 200, 80), width=2, point_radius=1)
        draw_lines_on_image(
            draw, [points for _, _, _, points in pred_items],
            img.size, pc_range, raw_xy,
            fill=(255, 255, 255), width=3, point_radius=2)
        if gt_lines is not None:
            draw_lines_on_image(
                draw, gt_lines, img.size, pc_range, raw_xy,
                fill=(255, 165, 0), width=2, point_radius=0)
        img.save(out_path)

    def save_pred_csv(csv_path, pred_items, lidar2global=None, map_location=None):
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'id', 'type', 'score',
                'wkt_local', 'points_local',
                'wkt_global', 'points_global',
                'wkt_wgs84', 'points_wgs84'])
            for pred_id, pred_type, pred_score, points in pred_items:
                global_points = (
                    local_to_global_points(points, lidar2global)
                    if lidar2global is not None else points)
                wgs84_points = global_to_wgs84_points(
                    global_points, map_location)
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
                wgs84_points = global_to_wgs84_points(
                    global_points, map_location)
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


    all_sample_dirs = []
    for i, data in enumerate(data_loader):
        if vis_count >= max_samples:
            break
        gt_labels_3d = batch_list(data['gt_labels_3d'])
        if len(gt_labels_3d) == 0 or ~(gt_labels_3d[0] != -1).any():
            # import pdb;pdb.set_trace()
            logger.error(f'\n empty gt for index {i}, continue')
            # prog_bar.update()  
            continue
       
        
        img_metas = meta_list(data['img_metas'])
        gt_bboxes_3d = batch_list(data['gt_bboxes_3d'])

        meta = img_metas[0]
        lidar2global = np.asarray(
            meta.get('lidar2global', np.eye(4)), dtype=np.float32)
        map_location = meta.get('map_location', None)
        raw_xy = bool(meta.get('raw_xy', False))
        if raw_xy:
            pts_filename = meta['sample_idx']
        else:
            pts_filename = meta['pts_filename']
        pts_filename = osp.basename(str(pts_filename))
        pts_filename = pts_filename.replace('__LIDAR_TOP__', '_').split('.')[0]
        if raw_xy:
            frame_idx = meta['frame_idx']
            pts_filename = f'{int(frame_idx):06d}_{pts_filename}'
        # import pdb;pdb.set_trace()
        # if pts_filename not in CANDIDATE:
        #     continue
        sample_dir = osp.join(args.show_dir, pts_filename)
        mmcv.mkdir_or_exist(osp.abspath(sample_dir))
        all_sample_dirs.append(sample_dir)
        print(len(all_sample_dirs))

        if args.meta_root is not None:
            meta_path = pts_filename.split('_',1)[1]
            meta_path = osp.join(args.meta_root, meta_path, "meta.json")
            if not osp.exists(meta_path):
                continue
            
        gt_raster_lines = gt_lines_from_bboxes(gt_bboxes_3d[0])

        with torch.no_grad():
            result = model(return_loss=False, rescale=True, **data)
        

        raster_input_path = None
        filename_list = img_metas[0].get('filename', [])
        if isinstance(filename_list, str):
            raster_input_path = osp.join(sample_dir, 'raster_input.png')
            if osp.exists(filename_list):
                shutil.copyfile(filename_list, raster_input_path)
            else:
                raster_input_path = None
            filename_list = []
        if filename_list:
            img_path_dict = {}
            # save cam img for sample
            for filepath in filename_list:
                cam_name = osp.basename(osp.dirname(filepath))
                img_name = cam_name + '.jpg'
                img_path = osp.join(sample_dir, img_name)
                shutil.copyfile(filepath, img_path)
                img_path_dict[cam_name] = img_path
            
            # surrounding view
            cam_names = list(img_path_dict.keys())
            split_idx = (len(cam_names) + 1) // 2
            row_1_list = [cv2.imread(img_path_dict[cam]) for cam in cam_names[:split_idx]]
            row_2_list = [cv2.imread(img_path_dict[cam]) for cam in cam_names[split_idx:]]
            row_1_list = [img for img in row_1_list if img is not None]
            row_2_list = [img for img in row_2_list if img is not None]
            if row_1_list or row_2_list:
                target_height = min(img.shape[0] for img in row_1_list + row_2_list)
                row_1_list = [
                    cv2.resize(img, (int(img.shape[1] * target_height / img.shape[0]), target_height))
                    for img in row_1_list
                ]
                row_2_list = [
                    cv2.resize(img, (int(img.shape[1] * target_height / img.shape[0]), target_height))
                    for img in row_2_list
                ]
                row_1_img=cv2.hconcat(row_1_list) if row_1_list else cv2.hconcat(row_2_list)
                row_2_img=cv2.hconcat(row_2_list) if row_2_list else row_1_img
                max_width = max(row_1_img.shape[1], row_2_img.shape[1])
                if row_1_img.shape[1] < max_width:
                    row_1_img = cv2.copyMakeBorder(
                        row_1_img, 0, 0, 0, max_width - row_1_img.shape[1],
                        cv2.BORDER_CONSTANT, value=0)
                if row_2_img.shape[1] < max_width:
                    row_2_img = cv2.copyMakeBorder(
                        row_2_img, 0, 0, 0, max_width - row_2_img.shape[1],
                        cv2.BORDER_CONSTANT, value=0)
                cams_img = cv2.vconcat([row_1_img,row_2_img])
                cams_img_path = osp.join(sample_dir,'surroud_view.jpg')
                cv2.imwrite(cams_img_path, cams_img,[cv2.IMWRITE_JPEG_QUALITY, 70])
        
        for vis_format in args.gt_format:
            if vis_format == 'se_pts':
                gt_line_points = gt_bboxes_3d[0].start_end_points
                for gt_bbox_3d, gt_label_3d in zip(gt_line_points, gt_labels_3d[0]):
                    pts = gt_bbox_3d.reshape(-1,2).numpy()
                    x = np.array([pt[0] for pt in pts])
                    y = np.array([pt[1] for pt in pts])
                    plt.quiver(x[:-1], y[:-1], x[1:] - x[:-1], y[1:] - y[:-1], scale_units='xy', angles='xy', scale=1, color=colors_plt[gt_label_3d])
            elif vis_format == 'bbox':
                gt_lines_bbox = gt_bboxes_3d[0].bbox
                for gt_bbox_3d, gt_label_3d in zip(gt_lines_bbox, gt_labels_3d[0]):
                    gt_bbox_3d = gt_bbox_3d.numpy()
                    xy = (gt_bbox_3d[0],gt_bbox_3d[1])
                    width = gt_bbox_3d[2] - gt_bbox_3d[0]
                    height = gt_bbox_3d[3] - gt_bbox_3d[1]
                    # import pdb;pdb.set_trace()
                    plt.gca().add_patch(Rectangle(xy,width,height,linewidth=0.4,edgecolor=colors_plt[gt_label_3d],facecolor='none'))
                    # plt.Rectangle(xy, width, height,color=colors_plt[gt_label_3d])
                # continue
            elif vis_format == 'fixed_num_pts':
                plt.figure(figsize=(2, 4))
                plt.xlim(pc_range[0], pc_range[3])
                plt.ylim(pc_range[1], pc_range[4])
                plt.axis('off')
                # gt_bboxes_3d[0].fixed_num=30 #TODO, this is a hack
                gt_lines_fixed_num_pts = gt_bboxes_3d[0].fixed_num_sampled_points
                for gt_bbox_3d, gt_label_3d in zip(gt_lines_fixed_num_pts, gt_labels_3d[0]):
                    # import pdb;pdb.set_trace() 
                    pts = gt_bbox_3d.numpy()
                    x = np.array([pt[0] for pt in pts])
                    y = np.array([pt[1] for pt in pts])
                    # plt.quiver(x[:-1], y[:-1], x[1:] - x[:-1], y[1:] - y[:-1], scale_units='xy', angles='xy', scale=1, color=colors_plt[gt_label_3d])

                    
                    plt.plot(x, y, color=colors_plt[gt_label_3d],linewidth=1,alpha=0.8,zorder=-1)
                    plt.scatter(x, y, color=colors_plt[gt_label_3d],s=2,alpha=0.8,zorder=-1)
                    # plt.plot(x, y, color=colors_plt[gt_label_3d])
                    # plt.scatter(x, y, color=colors_plt[gt_label_3d],s=1)
                if car_img is not None:
                    plt.imshow(car_img, extent=[-1.2, 1.2, -1.5, 1.5])
                else:
                    plt.scatter([0], [0], marker='s', s=12, color='deeppink')

                gt_fixedpts_map_path = osp.join(sample_dir, 'GT_fixednum_pts_MAP.png')
                plt.savefig(gt_fixedpts_map_path, bbox_inches='tight', format='png',dpi=1200)
                plt.close()   
            elif vis_format == 'polyline_pts':
                plt.figure(figsize=(2, 4))
                plt.xlim(pc_range[0], pc_range[3])
                plt.ylim(pc_range[1], pc_range[4])
                plt.axis('off')
                gt_lines_instance = gt_bboxes_3d[0].instance_list
                # import pdb;pdb.set_trace()
                for gt_line_instance, gt_label_3d in zip(gt_lines_instance, gt_labels_3d[0]):
                    pts = np.array(list(gt_line_instance.coords))
                    x = np.array([pt[0] for pt in pts])
                    y = np.array([pt[1] for pt in pts])
                    
                    # plt.quiver(x[:-1], y[:-1], x[1:] - x[:-1], y[1:] - y[:-1], scale_units='xy', angles='xy', scale=1, color=colors_plt[gt_label_3d])

                    # plt.plot(x, y, color=colors_plt[gt_label_3d])
                    plt.plot(x, y, color=colors_plt[gt_label_3d],linewidth=1,alpha=0.8,zorder=-1)
                    plt.scatter(x, y, color=colors_plt[gt_label_3d],s=1,alpha=0.8,zorder=-1)
                if car_img is not None:
                    plt.imshow(car_img, extent=[-1.2, 1.2, -1.5, 1.5])
                else:
                    plt.scatter([0], [0], marker='s', s=12, color='deeppink')

                gt_polyline_map_path = osp.join(sample_dir, 'GT_polyline_pts_MAP.png')
                plt.savefig(gt_polyline_map_path, bbox_inches='tight', format='png',dpi=1200)
                plt.close()           

            else: 
                logger.error(f'WRONG visformat for GT: {vis_format}')
                raise ValueError(f'WRONG visformat for GT: {vis_format}')


        # visualize pred
        # import pdb;pdb.set_trace()
        result_dic = result[0]['pts_bbox']
        boxes_3d = result_dic['boxes_3d'] # bbox: xmin, ymin, xmax, ymax
        scores_3d = result_dic['scores_3d']
        labels_3d = result_dic['labels_3d']
        pts_3d = result_dic['pts_3d']
        keep = scores_3d > args.score_thresh

        plt.figure(figsize=(2, 4))
        plt.xlim(pc_range[0], pc_range[3])
        plt.ylim(pc_range[1], pc_range[4])
        plt.axis('off')
        pred_items = []
        for pred_id, (pred_score_3d, pred_bbox_3d, pred_label_3d, pred_pts_3d) in enumerate(zip(scores_3d[keep], boxes_3d[keep],labels_3d[keep], pts_3d[keep])):

            pred_pts_3d = pred_pts_3d.cpu().numpy()
            pred_label_int = int(pred_label_3d.item())
            pred_score_float = float(pred_score_3d.item())
            pred_type = label_name(pred_label_int)
            pred_items.append((
                pred_id,
                pred_type,
                pred_score_float,
                pred_pts_3d.copy()))
            pts_x = pred_pts_3d[:,0]
            pts_y = pred_pts_3d[:,1]
            pred_color = colors_plt[pred_label_int % len(colors_plt)]
            plt.plot(pts_x, pts_y, color=pred_color,linewidth=1,alpha=0.8,zorder=-1)
            plt.scatter(pts_x, pts_y, color=pred_color,s=1,alpha=0.8,zorder=-1)


            pred_bbox_3d = pred_bbox_3d.cpu().numpy()
            xy = (pred_bbox_3d[0],pred_bbox_3d[1])
            width = pred_bbox_3d[2] - pred_bbox_3d[0]
            height = pred_bbox_3d[3] - pred_bbox_3d[1]
            pred_score_3d = float(pred_score_3d.item())
            pred_score_3d = round(pred_score_3d, 2)
            s = str(pred_score_3d)

        if car_img is not None:
            plt.imshow(car_img, extent=[-1.2, 1.2, -1.5, 1.5])
        else:
            plt.scatter([0], [0], marker='s', s=12, color='deeppink')

        map_path = osp.join(sample_dir, 'PRED_MAP_plot.png')
        plt.savefig(map_path, bbox_inches='tight', format='png',dpi=1200)
        plt.close()

        gt_rows = build_gt_rows(
            meta.get('local_annotation', {}),
            meta.get('global_annotation', {}),
            lidar2global)

        # save_pred_csv(
        #     osp.join(sample_dir, 'pred_vectors.csv'),
        #     pred_items,
        #     lidar2global=lidar2global,
        #     map_location=map_location)
        # save_gt_global_csv(
        #     osp.join(sample_dir, 'gt_global_map.csv'),
        #     gt_rows,
        #     map_location=map_location)

        save_pred_csv_xd(pts_filename, 
                         osp.join(sample_dir, "pred_wgs84.csv"), 
                         meta_path, pred_items)
        save_gt_csv_xd(pts_filename, 
                       osp.join(sample_dir, "gt_wgs84.csv"), 
                       meta_path, gt_rows)
        
        
        trajectory = None
        source_npz_path = meta.get('source_npz_path', None)
        if source_npz_path and osp.exists(source_npz_path):
            try:
                source_npz = np.load(source_npz_path, allow_pickle=True)
                if 'trajectory' in source_npz:
                    trajectory = np.asarray(source_npz['trajectory'], dtype=np.float32)
            except Exception as e:
                print(f'WARN: failed to load trajectory from {source_npz_path}: {e}')
        draw_pred_on_raster(
            raster_input_path,
            osp.join(sample_dir, 'PRED_on_raster.png'),
            pred_items,
            pc_range,
            raw_xy=raw_xy,
            trajectory=trajectory)
        draw_pred_on_raster(
            raster_input_path,
            osp.join(sample_dir, 'GT_PRED_on_raster.png'),
            pred_items,
            pc_range,
            raw_xy=raw_xy,
            gt_lines=gt_raster_lines,
            trajectory=trajectory)

        
        prog_bar.update()
        vis_count += 1

    all_pred_csv_path = osp.join(args.show_dir, 'all_pred_wgs84.csv')
    merge_all_xd_csv(all_sample_dirs, 'pred_wgs84.csv', all_pred_csv_path)
    
    all_gt_csv_path = osp.join(args.show_dir,'all_gt_wgs84.csv')
    merge_all_xd_csv(all_sample_dirs, 'gt_wgs84.csv',  all_gt_csv_path)

    logger.info('\n DONE vis test dataset samples gt label & pred')

import debugpy
if 0:
	debugpy.listen(("0.0.0.0", 5678))
	print("🔥 waiting for debugger attach...")
	debugpy.wait_for_client()

if __name__ == '__main__':
    main()
