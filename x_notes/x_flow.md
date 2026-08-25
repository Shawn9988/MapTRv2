# MapTRv2 Mini 跑通流程

这份文档讲“要干什么、怎么判断做对了”。具体可复制命令放在
`x_run_nus.md` 里，优先看其中的 `Mini Overfit` 一节。

## 0. 当前链路

当前只用 244：

```text
本地改代码:
E:/0hn/2code/22servercode/MapTR

SFTP 自动同步到:
huinian@10.130.21.244:/data/huinian/source/MapTR/

SSH 到 244 运行:
/data/huinian/source/MapTR
```

判断是否正确：

```text
.vscode/sftp.json 指向 10.130.21.244
SSH 后 pwd 是 /data/huinian/source/MapTR
不要去 214，也不要去 /nas/nfs/large-model/hn/code/MapTR
```

## 1. 进入运行环境

目的：进入已经配好的 `py38` 环境，并让 Python 能 import 当前项目代码。

怎么做：按 `x_run_nus.md` 开头的环境命令执行。

做对了应该看到：

```text
机器是 10.130.21.244
当前目录是 /data/huinian/source/MapTR
conda 环境是 py38
```

常用检查：

```bash
hostname -I
pwd
```

## 2. 生成 Mini info

目的：把 nuScenes Mini 数据集整理成 MapTRv2 能读取的索引文件。

怎么做：执行 `x_run_nus.md` 里 `Build nuScenes mini infos` 的命令。

做对了应该生成：

```text
data/nuscenes/nuscenes_mini_map_infos_temporal_train.pkl
data/nuscenes/nuscenes_mini_map_infos_temporal_val.pkl
```

如果这两个文件已经存在，并且没有改数据集，可以不重复生成。

## 3. 导出 Mini raster 样本

目的：把 Mini 的地图 boundary 和雷达信息转成 raster 训练样本。

怎么做：执行 `x_run_nus.md` 里 `Export mini train raster samples` 的命令。

当前 overfit config 默认读取：

```text
work_dirs/nuscenes_radar_map_samples_train_all/maptr_raster_infos.pkl
```

所以 Mini raster 也导出到这个目录，后面的训练和可视化就不用额外改 config。

做对了应该生成：

```text
work_dirs/nuscenes_radar_map_samples_train_all/maptr_raster_infos.pkl
work_dirs/nuscenes_radar_map_samples_train_all/maptr_raster_infos.json
work_dirs/nuscenes_radar_map_samples_train_all/samples/*.png
work_dirs/nuscenes_radar_map_samples_train_all/labels/*.png
work_dirs/nuscenes_radar_map_samples_train_all/*.npz
```

提醒：如果这个目录里原来有数据，重新 export 会替换/混入新样本。只是想用
已有数据训练的话，可以跳过第 2 步和第 3 步，直接训练。

## 4. 训练 Mini overfit

目的：用 Mini raster 样本训练 boundary 预测模型，先确认完整流程能跑通。

怎么做：执行 `x_run_nus.md` 里 `Train` 的 overfit 命令。

使用的 config 是：

```text
projects/configs/maptrv2/maptrv2_nusc_raster_boundary_overfit.py
```

做对了应该出现训练目录：

```text
work_dirs/maptrv2_nusc_raster_boundary_overfit/
```

里面通常会有：

```text
latest.pth
epoch_*.pth
*.log
*.log.json
```

## 5. 可视化 Mini 结果

目的：快速看模型预测出来的 boundary 是否有形状、是否和输入区域对应。

怎么做：执行 `x_run_nus.md` 里 `Visualize train samples` 的命令。

做对了应该出现可视化目录：

```text
work_dirs/maptrv2_nusc_raster_boundary_overfit/vis_train/
```

里面应该有预测结果图片。

## 6. 常见检查

代码改了但运行没变化：

```bash
cd /data/huinian/source/MapTR
grep -R "你刚改过的代码标记" -n tools projects
```

训练找不到数据：

```bash
ls -lh work_dirs/nuscenes_radar_map_samples_train_all/maptr_raster_infos.pkl
```

可视化找不到模型：

```bash
ls -lh work_dirs/maptrv2_nusc_raster_boundary_overfit/latest.pth
```

## 7. 2k 流程

2k 流程现在不讲，保留在 `x_run_nus.md` 里。只有明确要从 Mini 切到 2k
时再用：

```text
projects/configs/maptrv2/maptrv2_nusc_raster_boundary_2k.py
projects/configs/maptrv2/maptrv2_nusc_raster_boundary_2k_48ep.py
```

## 总结

训练先跑 nuScenes Mini。环境已经配好，服务器，进 `/data/huinian/source/MapTR`，激活 `py38`，按 `x_run_nus.md` 的 `Mini Overfit` 依次生成 info、导出 raster、训练、可视化。
