nuScenes radar raster -> MapTRv2 vector boundary 路线

🌲路线树：
nuScenes 原始数据
  -> MapTR temporal infos
  -> radar raster samples + vector GT
  -> CustomRasterMapDataset
  -> raster encoder
  -> MapTRv2 vector decoder
  -> boundary polyline / WKT

🌲关键代码结构树：
MapTR
├── x_run_nus.md
│   └── 记录实验命令：数据导出、训练、画 loss、可视化预测
│
├── tools
│   ├── maptrv2
│   │   ├── custom_nusc_map_converter.py
│   │   │   └── nuScenes 原始 metadata -> MapTR infos pkl
│   │   │       输出：
│   │   │       ├── data/nuscenes/nuscenes_2k_map_infos_temporal_train.pkl
│   │   │       └── data/nuscenes/nuscenes_2k_map_infos_temporal_val.pkl
│   │   │
│   │   └── export_nusc_radar_map_samples.py
│   │       └── MapTR infos + radar sweeps + map vector
│   │           -> raster 输入图 + vector GT + 检查图
│   │           输出：
│   │           ├── work_dirs/nuscenes_2k_radar_map_samples_train/
│   │           │   ├── sample_xxx/
│   │           │   │   ├── input.png              # 灰度输入图
│   │           │   │   ├── label.png / check.png  # 检查用图
│   │           │   │   └── sample.npz
│   │           │   ├── maptr_raster_infos.pkl
│   │           │   └── maptr_raster_infos.json
│   │           │
│   │           └── work_dirs/nuscenes_2k_radar_map_samples_val/
│   │               └── 同上
│   │
│   ├── train.py
│   │   └── 训练入口
│   │
│   ├── dist_train.sh
│   │   └── 分布式/单卡训练包装脚本
│   │
│   └── maptr
│       └── vis_pred.py
│           └── checkpoint -> 预测矢量可视化 / CSV / WKT 导出
│
├── projects
│   ├── configs
│   │   └── maptrv2
│   │       ├── maptrv2_nusc_raster_boundary_overfit.py
│   │       │   └── 小样本 overfit 配置
│   │       │
│   │       └── maptrv2_nusc_raster_boundary_2k.py
│   │           └── 当前 2k 训练配置
│   │               数据：1520 train + 480 val
│   │               epoch：24
│   │               checkpoint：每 6 epoch 存一次
│   │
│   └── mmdet3d_plugin
│       ├── datasets
│       │   ├── raster_map_dataset.py
│       │   │   └── CustomRasterMapDataset
│       │   │       读取 maptr_raster_infos.pkl
│       │   │       输出 raster image + vector GT
│       │   │
│       │   └── pipelines
│       │       └── loading.py
│       │           └── LoadRasterMapImage
│       │               读取灰度 input.png
│       │
│       └── maptr
│           ├── detectors
│           │   └── maptrv2.py
│           │       └── MapTRv2 主模型
│           │           支持 modality='raster'
│           │
│           └── modules
│               └── transformer.py
│                   └── MapTR transformer / vector decoder
│
└── work_dirs
    └── maptrv2_nusc_raster_boundary_2k
        ├── *.log / *.log.json
        ├── epoch_6.pth
        ├── epoch_12.pth
        ├── epoch_18.pth
        ├── epoch_24.pth
        ├── latest.pth
        └── loss_final_curve.png


🌲模型内部扭转：
input.png 灰度 raster
    ↓
LoadRasterMapImage
    ↓
CustomRasterMapDataset
    ↓
SimpleRasterEncoder
    ↓
BEV feature
    ↓
MapTRv2 Transformer Decoder
    ↓
Vector Head
    ├── class score
    ├── polyline points
    └── direction loss
    ↓
pred boundary vectors