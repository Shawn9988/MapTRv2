#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""校验 XD 训练管线里的几何增广：把增广后的 raster_img 和增广后的 GT 一起画出来。

check_xd_maptr_infos.py 读的是 pkl 静态 GT，看不到增广；增广是训练时在 pipeline
里实时做的。本脚本直接实例化 train 数据集（含 XDMapGeometricAug），取增广后的样本
overlay 保存，逐张确认 图 与 GT 仍然对齐。

用法（从仓库根目录运行）：
  python tools/maptrv2/check_xd_aug.py \
      projects/configs/maptrv2/maptrv2_xd_raster_boundary.py \
      --num 8 --repeat 3 --out-dir work_dirs/aug_check
每个样本重复取 repeat 次（每次随机增广不同），便于看变换范围。
"""

import argparse
import os
import os.path as osp

import numpy as np
from mmcv import Config
from PIL import Image, ImageDraw

# 注册 plugin 里的 dataset / pipeline
import projects.mmdet3d_plugin  # noqa: F401
from mmdet.datasets import build_dataset


def parse_args():
    p = argparse.ArgumentParser(description='Check XD geometric augmentation')
    p.add_argument('config')
    p.add_argument('--num', type=int, default=8, help='samples to visualize')
    p.add_argument('--repeat', type=int, default=3,
                   help='times to re-sample each (different random aug)')
    p.add_argument('--out-dir', default='work_dirs/aug_check')
    return p.parse_args()


def tensor_to_img(tensor):
    """C,H,W float -> H,W,3 uint8 (仅供肉眼看，做个 0-1 裁剪缩放)。"""
    arr = tensor.numpy()
    arr = np.clip(arr, 0.0, 1.0)
    arr = (arr * 255).astype(np.uint8).transpose(1, 2, 0)
    if arr.shape[2] == 1:
        arr = np.repeat(arr, 3, axis=2)
    return arr


def main():
    args = parse_args()
    cfg = Config.fromfile(args.config)
    ds = build_dataset(cfg.data.train)
    pc = list(ds.pc_range)
    x_min, y_min, x_max, y_max = pc[0], pc[1], pc[3], pc[4]

    os.makedirs(args.out_dir, exist_ok=True)
    n = min(args.num, len(ds))
    print(f'dataset size={len(ds)}, visualizing {n} x {args.repeat}')
    for i in range(n):
        for r in range(args.repeat):
            ex = ds[i]                       # 走完整 train pipeline（含增广）
            img = ex['raster_img'].data      # 增广后的图
            H, W = img.shape[1], img.shape[2]
            res_x = (x_max - x_min) / (W - 1)
            res_y = (y_max - y_min) / (H - 1)

            pil = Image.fromarray(tensor_to_img(img))
            draw = ImageDraw.Draw(pil)
            # 增广后的 GT（米制，已居中）
            for line in ex['gt_bboxes_3d'].data.instance_list:
                coords = np.array(list(line.coords))
                pts = [((x - x_min) / res_x, (y_max - y) / res_y)
                       for x, y in coords[:, :2]]
                if len(pts) >= 2:
                    draw.line(pts, fill=(255, 0, 255), width=2)
            pil.save(osp.join(args.out_dir, f'{i:03d}_{r}.png'))
    print('done ->', args.out_dir, '（品红=增广后的 GT，应压在增广后的图上）')


if __name__ == '__main__':
    main()
