import argparse
import os
import os.path as osp

import numpy as np
from PIL import Image


def parse_args():
    parser = argparse.ArgumentParser(
        description='Convert one vendor screenshot/map image to MapTR grayscale raster sample.')
    parser.add_argument('--image', required=True, help='input screenshot/map image')
    parser.add_argument('--out', required=True, help='output grayscale sample png')
    parser.add_argument(
        '--crop',
        type=int,
        nargs=4,
        default=None,
        metavar=('LEFT', 'TOP', 'RIGHT', 'BOTTOM'),
        help='optional pixel crop box before conversion')
    parser.add_argument(
        '--image-size',
        type=int,
        nargs=2,
        default=None,
        metavar=('WIDTH', 'HEIGHT'),
        help='optional output size')
    parser.add_argument(
        '--keep-dark',
        action='store_true',
        help='keep dark/black structures as radar-like pixels')
    return parser.parse_args()


def main():
    args = parse_args()
    img = Image.open(args.image).convert('RGB')
    if args.crop is not None:
        img = img.crop(tuple(args.crop))
    if args.image_size is not None:
        img = img.resize(tuple(args.image_size), Image.BILINEAR)

    rgb = np.asarray(img).astype(np.int16)
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]

    out = np.full(rgb.shape[:2], 255, dtype=np.uint8)

    # Match the single-channel training raster convention:
    # radar/dark points: 0, divider/lane prior: 80, trajectory: 150, ego marker: 200.
    blue = (b > 120) & (b > r + 40) & (b > g + 20)
    purple = (r > 100) & (b > 100) & (g < 140)
    green = (g > 120) & (g > r + 30) & (g > b + 20)
    dark = (r < 80) & (g < 80) & (b < 80)

    out[blue] = 80
    out[green] = 150
    out[purple] = 200
    if args.keep_dark:
        out[dark] = 0

    os.makedirs(osp.dirname(osp.abspath(args.out)), exist_ok=True)
    Image.fromarray(out, mode='L').save(args.out)
    print(f'Wrote grayscale raster sample to {args.out}')


if __name__ == '__main__':
    main()
