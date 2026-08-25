# MapTRv2 AV2 Run Log

This file keeps only the original AV2 image-based reproduction path. Temporary
AV2 lidar/fusion debug configs were removed during config cleanup.

## Main Configs

```text
projects/configs/maptrv2/maptrv2_av2_3d_r50_6ep.py
projects/configs/maptrv2/maptrv2_av2_3d_r50_6ep_w_centerline.py
```

## Preprocess AV2

```bash
cd /data/huinian/source/MapTR
python tools/maptrv2/custom_av2_map_converter.py \
  --data-root ./data/argoverse2/sensor/ \
  --nproc 4
```

Expected outputs:

```text
data/argoverse2/sensor/av2_map_infos_train.pkl
data/argoverse2/sensor/av2_map_infos_val.pkl
data/argoverse2/sensor/av2_map_infos_test.pkl
```

## Smoke Train

```bash
cd /data/huinian/source/MapTR
chmod +x tools/dist_train.sh

bash tools/dist_train.sh \
  projects/configs/maptrv2/maptrv2_av2_3d_r50_6ep.py \
  1 \
  --cfg-options data.samples_per_gpu=1 data.workers_per_gpu=0 total_epochs=1 runner.max_epochs=1
```

## Eval

```bash
bash tools/dist_test_map.sh \
  projects/configs/maptrv2/maptrv2_av2_3d_r50_6ep.py \
  work_dirs/maptrv2_av2_3d_r50_6ep/latest.pth \
  1 \
  --cfg-options data.samples_per_gpu=1 data.workers_per_gpu=0
```

## Visualization

```bash
cd /data/huinian/source/MapTR
export PYTHONPATH="/data/huinian/source/MapTR:$PYTHONPATH"

python tools/maptr/vis_pred.py \
  projects/configs/maptrv2/maptrv2_av2_3d_r50_6ep.py \
  work_dirs/maptrv2_av2_3d_r50_6ep/latest.pth \
  --show-dir work_dirs/maptrv2_av2_3d_r50_6ep/vis_pred \
  --score-thresh 0.05
```
