## nuScenes Raster MapTRv2

```bash
conda activate py38
cd /data/huinian/source/MapTR
export PYTHONPATH="/data/huinian/source/MapTR:$PYTHONPATH"
```

### Mini Overfit

The overfit config name is kept unchanged. It reads raster samples from
`work_dirs/nuscenes_radar_map_samples_train_all/maptr_raster_infos.pkl`.

Build nuScenes mini infos:

```bash
python tools/maptrv2/custom_nusc_map_converter.py \
  --root-path ./data/nuscenes \
  --out-dir ./data/nuscenes \
  --extra-tag nuscenes_mini \
  --version v1.0-mini \
  --canbus ./data/nuscenes \
  --max-sweeps 10
```

Expected outputs:

```text
data/nuscenes/nuscenes_mini_map_infos_temporal_train.pkl
data/nuscenes/nuscenes_mini_map_infos_temporal_val.pkl
```

Export mini train raster samples for the overfit config:

```bash
python tools/maptrv2/export_nusc_radar_map_samples.py \
  --info data/nuscenes/nuscenes_mini_map_infos_temporal_train.pkl \
  --out-dir work_dirs/nuscenes_radar_map_samples_train_all \
  --start-index 0 \
  --num-samples 323 \
  --sweeps-num 10 \
  --pc-range -30 -40 30 40 \
  --image-size 800 600
```

Train:

```bash
bash tools/dist_train.sh \
  projects/configs/maptrv2/maptrv2_nusc_raster_boundary_overfit.py \
  1
```

Visualize train samples:

```bash
python tools/maptr/vis_pred.py \
  projects/configs/maptrv2/maptrv2_nusc_raster_boundary_overfit.py \
  work_dirs/maptrv2_nusc_raster_boundary_overfit/latest.pth \
  --show-dir work_dirs/maptrv2_nusc_raster_boundary_overfit/vis_train \
  --score-thresh 0.03 \
  --max-samples 8
```

### 2k Dataset

Build 2k nuScenes infos:

```bash
python tools/maptrv2/custom_nusc_map_converter.py \
  --root-path ./data/nuscenes \
  --out-dir ./data/nuscenes \
  --extra-tag nuscenes_2k \
  --version v1.0-trainval \
  --canbus ./data/nuscenes \
  --max-sweeps 10 \
  --max-samples 2000
```

Export train rasters:

```bash
python tools/maptrv2/export_nusc_radar_map_samples.py \
  --info data/nuscenes/nuscenes_2k_map_infos_temporal_train.pkl \
  --out-dir work_dirs/nuscenes_2k_radar_map_samples_train \
  --start-index 0 \
  --num-samples 1520 \
  --sweeps-num 10 \
  --pc-range -30 -40 30 40 \
  --image-size 800 600
```

Export val rasters:

```bash
python tools/maptrv2/export_nusc_radar_map_samples.py \
  --info data/nuscenes/nuscenes_2k_map_infos_temporal_val.pkl \
  --out-dir work_dirs/nuscenes_2k_radar_map_samples_val \
  --start-index 0 \
  --num-samples 480 \
  --sweeps-num 10 \
  --pc-range -30 -40 30 40 \
  --image-size 800 600
```

Train 24 epochs:

```bash
bash tools/dist_train.sh \
  projects/configs/maptrv2/maptrv2_nusc_raster_boundary_2k.py \
  1
```

Train or resume 48 epochs:

```bash
bash tools/dist_train.sh \
  projects/configs/maptrv2/maptrv2_nusc_raster_boundary_2k_48ep.py \
  1
```

Plot loss:

```bash
python tools/analysis_tools/analyze_logs.py plot_curve \
  work_dirs/maptrv2_nusc_raster_boundary_2k/*.log.json \
  --keys loss \
  --legend loss \
  --out work_dirs/maptrv2_nusc_raster_boundary_2k/loss_total_curve.png

python tools/analysis_tools/analyze_logs.py plot_curve \
  work_dirs/maptrv2_nusc_raster_boundary_2k/*.log.json \
  --keys loss_pts loss_dir loss_cls \
  --legend loss_pts loss_dir loss_cls \
  --out work_dirs/maptrv2_nusc_raster_boundary_2k/loss_final_curve.png
```

Visualize 24-epoch checkpoint:

```bash
python tools/maptr/vis_pred.py \
  projects/configs/maptrv2/maptrv2_nusc_raster_boundary_2k.py \
  work_dirs/maptrv2_nusc_raster_boundary_2k/epoch_24.pth \
  --show-dir work_dirs/maptrv2_nusc_raster_boundary_2k/vis_epoch_24 \
  --score-thresh 0.2 \
  --max-samples 32
```

Visualize 48-epoch checkpoint:

```bash
python tools/maptr/vis_pred.py \
  projects/configs/maptrv2/maptrv2_nusc_raster_boundary_2k.py \
  work_dirs/maptrv2_nusc_raster_boundary_2k_48ep/epoch_48.pth \
  --show-dir work_dirs/maptrv2_nusc_raster_boundary_2k_48ep/vis_epoch_48 \
  --score-thresh 0.2 \
  --max-samples 32
```

## xd MapTR Samples

Convert xd train/val samples to MapTR raster infos:

Expected input folder:

```text
/data/huinian/data/MapTR/train_samples/*/raster.png
/data/huinian/data/MapTR/train_samples/*/raster_with_boundary.png
/data/huinian/data/MapTR/train_samples/*/sample.json
/data/huinian/data/MapTR/val_samples/*/raster.png
/data/huinian/data/MapTR/val_samples/*/raster_with_boundary.png
/data/huinian/data/MapTR/val_samples/*/sample.json
```

```bash
python tools/maptrv2/build_xd_maptr_infos.py \
  --sample-root /data/huinian/data/MapTR/train_samples \
  --out-dir work_dirs/xd_maptr_samples_train \
  --pc-range 0 -40 100 40

python tools/maptrv2/build_xd_maptr_infos.py \
  --sample-root /data/huinian/data/MapTR/val_samples \
  --out-dir work_dirs/xd_maptr_samples_val \
  --pc-range 0 -40 100 40
```

python tools/analysis_tools/analyze_logs.py plot_curve \
  work_dirs/maptrv2_xd_raster_boundary/*.log.json \
  --keys loss \
  --legend loss \
  --out work_dirs/maptrv2_xd_raster_boundary/loss_total_curve.png

Train xd raster model:

```text
raster.png channel convention:
  B = radar points
  G = lane / lane boundary
  R = ego trajectory
```

```bash
bash tools/dist_train.sh \
  projects/configs/maptrv2/maptrv2_xd_raster_boundary.py \
  1
```

bash tools/dist_train.sh \
  projects/configs/maptrv2/maptrv2_xd_raster_boundary.py \
  1 \
  --resume-from work_dirs/maptrv2_xd_raster_boundary/epoch_24.pth

Visualize xd checkpoint:

```bash
python -u tools/maptr/vis_pred.py \
  projects/configs/maptrv2/maptrv2_xd_raster_boundary.py \
  work_dirs/maptrv2_xd_raster_boundary/epoch_600.pth \
  --show-dir work_dirs/maptrv2_xd_raster_boundary/vis \
  --score-thresh 0.05 \
  --max-samples 32
```
