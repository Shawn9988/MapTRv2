#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""校验 build_xd_maptr_infos.py 产出的 maptr_raster_infos.pkl 是否正常。

两层检查：
  1) 数值范围：统计所有 boundary 点的 (x, y) 极值，并数出落在 point_cloud_range
     之外的点。正常情况下 x∈[pc_x_min, pc_x_max]、y∈[pc_y_min, pc_y_max]，越界数应为 0。
     （之前 backfill 把 s 存成绝对值时，x 会飞到 [50,150]，这里会立刻暴露。）
  2) 可视化叠加：把 pkl 里的 GT boundary 用 MapTR 的同一套坐标约定反投影回 raster.png，
     若画出的线正好压在图里真实的道路边界上，说明转换正确、且与栅格图对齐。

用法：
  python tools/maptrv2/check_xd_maptr_infos.py \
      --pkl work_dirs/xd_maptr_samples_train/maptr_raster_infos.pkl \
      --out-dir work_dirs/xd_maptr_samples_train/check_vis \
      --num 20
"""

import argparse
import os
import os.path as osp

import mmcv
import numpy as np
from PIL import Image, ImageDraw


def parse_args():
    parser = argparse.ArgumentParser(description='Check xd maptr_raster_infos.pkl')
    parser.add_argument('--pkl', required=True, help='maptr_raster_infos.pkl path')
    parser.add_argument('--out-dir', default=None,
                        help='dir to dump overlay pngs (default: <pkl_dir>/check_vis)')
    parser.add_argument('--num', type=int, default=20,
                        help='number of samples to visualize')
    parser.add_argument('--classes', nargs='+',
                        default=['boundary'],
                        help='annotation keys to check/draw')
    return parser.parse_args()


# 每个类别一个颜色（RGB），画在 raster 上便于区分
CLASS_COLORS = {
    'boundary': (255, 0, 255),     # 品红
    'divider': (255, 255, 0),      # 黄
    'ped_crossing': (0, 255, 255),  # 青
    'centerline': (255, 128, 0),   # 橙
}


def sd_to_pixel(x, y, pc, W, H):
    """MapTR BEV 米制 (x, y) -> raster 像素。

    x∈[pc_x_min, pc_x_max] 映射到列 [0, W-1]，左列=pc_x_min；
    y∈[pc_y_min, pc_y_max] 映射到行 [0, H-1]，顶行=pc_y_max（与 C++ ToPixel 的
    py=(lateral_range - d)/res 一致）。
    """
    x_min, y_min, _, x_max, y_max, _ = pc
    res_x = (x_max - x_min) / max(1, (W - 1))
    res_y = (y_max - y_min) / max(1, (H - 1))
    px = (x - x_min) / res_x
    py = (y_max - y) / res_y
    return px, py


def main():
    args = parse_args()
    data = mmcv.load(args.pkl)
    infos = data['infos']
    meta = data.get('metadata', {})
    print(f'== loaded {len(infos)} samples from {args.pkl}')
    print(f'== metadata pc_range={meta.get("pc_range")} '
          f'x_offset={meta.get("x_offset")}')

    # ---- 1) 数值范围检查 ----
    gmin = np.array([np.inf, np.inf])
    gmax = np.array([-np.inf, -np.inf])
    total_pts = 0
    oob_pts = 0
    empty_samples = 0
    for info in infos:
        pc = info['pc_range']
        x_min, y_min, _, x_max, y_max, _ = pc
        has_pt = False
        for cls in args.classes:
            for line in info['annotation'].get(cls, []):
                arr = np.asarray(line, dtype=np.float64)
                if arr.ndim != 2 or arr.shape[0] == 0:
                    continue
                has_pt = True
                xy = arr[:, :2]
                gmin = np.minimum(gmin, xy.min(axis=0))
                gmax = np.maximum(gmax, xy.max(axis=0))
                total_pts += xy.shape[0]
                oob = ((xy[:, 0] < x_min) | (xy[:, 0] > x_max) |
                       (xy[:, 1] < y_min) | (xy[:, 1] > y_max))
                oob_pts += int(oob.sum())
        if not has_pt:
            empty_samples += 1

    print('\n-- range check --')
    print(f'   total boundary points : {total_pts}')
    print(f'   x range               : [{gmin[0]:.3f}, {gmax[0]:.3f}]')
    print(f'   y range               : [{gmin[1]:.3f}, {gmax[1]:.3f}]')
    print(f'   samples w/o any GT     : {empty_samples}')
    print(f'   out-of-range points    : {oob_pts}')
    if oob_pts == 0:
        print('   [OK] 所有 GT 点都在 point_cloud_range 内')
    else:
        pct = 100.0 * oob_pts / max(1, total_pts)
        print(f'   [FAIL] 有 {oob_pts} ({pct:.1f}%) 个点越界 —— 坐标变换很可能仍有问题')

    # ---- 2) 可视化叠加 ----
    out_dir = args.out_dir or osp.join(osp.dirname(args.pkl), 'check_vis')
    os.makedirs(out_dir, exist_ok=True)
    n = min(args.num, len(infos))
    print(f'\n-- writing {n} overlay images to {out_dir} --')
    for info in infos[:n]:
        raster_path = info['bev_img_path']
        if not osp.exists(raster_path):
            print(f'   [skip] raster missing: {raster_path}')
            continue
        img = Image.open(raster_path).convert('RGB')
        W, H = img.size
        draw = ImageDraw.Draw(img)
        pc = info['pc_range']
        for cls in args.classes:
            color = CLASS_COLORS.get(cls, (255, 255, 255))
            for line in info['annotation'].get(cls, []):
                arr = np.asarray(line, dtype=np.float64)
                if arr.ndim != 2 or arr.shape[0] < 2:
                    continue
                pts = [sd_to_pixel(x, y, pc, W, H) for x, y in arr[:, :2]]
                draw.line(pts, fill=color, width=2)
        out_path = osp.join(out_dir, f'{info["token"]}_check.png')
        img.save(out_path)
    print('done. 逐张打开确认：GT 线应压在 raster 里真实的道路边界上。')


if __name__ == '__main__':
    main()
