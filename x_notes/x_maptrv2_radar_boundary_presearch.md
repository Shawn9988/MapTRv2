# MapTRv2 用于毫米波道路边界提取的技术路线预研

## 1. 预研问题

本次预研要回答的核心问题不是：

```text
MapTRv2 是什么？
```

而是：

```text
Radar Point Cloud
    -> Vectorized Road Boundary

这条路线是否有谱，是否值得继续投入？
```

更具体地说，我们希望判断：

```text
能否借鉴 MapTR / MapTRv2 的 vector map 思路，
从毫米波点云中提取道路边界 polyline。
```

这不是单纯的论文复现问题，而是一个技术路线立项问题。组内真正需要判断的是：

- 为什么选择 MapTRv2 这类 vector map 框架？
- 当前有没有证据说明这不是拍脑袋？
- 已经验证到哪一步？
- 剩下最大的技术风险是什么？
- 是否值得继续投入人力做毫米波实验？

## 2. 当前结论

阶段性结论：

```text
MapTRv2 路线已具备继续开展毫米波验证的技术依据。
```

但需要注意：目前还不能说“毫米波方案已经可行”，因为我们尚未完成毫米波数据实验。

当前实验能够支撑的结论是：

```text
MapTRv2 路线具备继续投入毫米波验证的价值。
```

已验证：

```text
LiDAR Point Cloud
    -> BEV Feature
    -> MapTRv2 Decoder
    -> Boundary Polyline
```

这条链路已经验证成立。也就是说，点云 BEV 可以驱动 MapTRv2 风格的 vector decoder 学习道路边界结构。

因此，当前主要风险已经从：

```text
Vector Map 路线本身是否可行？
```

收敛到：

```text
毫米波点云能否构建出足够有效的 BEV feature？
```

换句话说，本次预研的价值不是证明“毫米波已经能提边界”，而是证明：

```text
MapTRv2 这条 vector boundary 路线值得继续往毫米波方向验证。
```

## 3. 证据链

当前完成了三阶段验证：

```text
Image
  -> MapTRv2
  -> Boundary / Vector Map
  ✓ 官方链路跑通

Image + LiDAR
  -> MapTRv2
  -> Boundary / Vector Map
  ✓ 点云 BEV 可以接入 MapTRv2Head

Pure LiDAR
  -> MapTRv2Head
  -> Boundary
  ✓ 训练集上已学出道路边界结构

Radar
  -> MapTRv2Head
  -> Boundary
  ? 下一阶段待验证
```

这条证据链说明：

```text
MapTRv2Head 不是只能服务 image 输入；
它可以从点云 BEV 中学习 vector boundary。
```

因此，后续毫米波路线可以抽象为：

```text
Radar Encoder
    -> Radar BEV Feature
    -> MapTRv2Head
    -> Boundary Polyline
```

当前工作的重点已经从：

```text
研究 MapTRv2
```

转变为：

```text
设计适合毫米波的 BEV Encoder。
```

本次预研验证的不只是 LiDAR boundary 提取能力，更重要的是验证了：

```text
Point Cloud
    -> BEV
    -> Vector Boundary
```

这一技术范式具备可行性。

对于公司现有地图生产流程而言，最终目标不是得到某一种传感器结果，而是得到可进入地图生产链路的 vector map。因此如果这条范式成立，未来 LiDAR、Image、Radar 或其他传感器理论上都可以共享统一的 Vector Map Decoder。

当前毫米波项目可以理解为这一技术路线在 Radar 场景下的首次落地验证。

## 4. 为什么先做 LiDAR 验证

如果一开始直接做：

```text
Radar
  -> Boundary
```

一旦失败，很难判断问题来源。可能是：

- Radar 点云太稀疏；
- 道路边界在 Radar 中不可观测；
- Radar BEV encoder 不合适；
- MapTRv2 不适合 boundary；
- 训练配置或数据链路有问题。

因此我们先引入 LiDAR 作为中间验证变量：

```text
LiDAR
  -> BEV
  -> MapTRv2Head
  -> Boundary
```

LiDAR 比 Radar 更密集、更稳定。如果 LiDAR 都无法通过 MapTRv2Head 学出 boundary，那么这条 vector map 路线本身就值得怀疑。

现在 LiDAR 验证已经成立，因此可以把问题拆开：

```text
已验证：
  BEV -> MapTRv2Head -> Boundary

待验证：
  Radar -> BEV
```

这一步的意义是降低技术风险：我们不是一次性证明完整 Radar 方案，而是先证明 vector decoder 路线值得继续走。

## 5. MapTRv2 在本项目中的角色

MapTR / MapTRv2 原本用于在线矢量化 HD map 构建。给定车端传感器输入，模型直接预测 ego 周围的地图元素，例如：

- lane divider；
- pedestrian crossing；
- road boundary；
- centerline，MapTRv2 中可选扩展。

它和传统 BEV segmentation 的主要区别是：

```text
传统路线：
  BEV feature
    -> raster segmentation
    -> 后处理提取边界线

MapTRv2 路线：
  BEV feature
    -> vector decoder
    -> polyline
```

对于道路边界提取，后者更贴近我们的目标，因为输出天然是结构化边界线：

```text
boundary = [(x1, y1), (x2, y2), ..., (xn, yn)]
```

在本项目中，MapTRv2 的定位不是完整端到端方案，而是：

```text
一个可复用的 Vector Map Decoder。
```

我们希望复用：

- map query 机制；
- polyline point prediction；
- prediction 与 GT polyline 的 matching；
- point set / direction 相关损失；
- boundary 这类矢量地图元素表达。

我们需要替换的是：

```text
原始 Image BEV Encoder
    -> Radar BEV Encoder
```

## 6. 为什么选择 MapTRv2

选择 MapTRv2 不是因为要复现论文，而是因为它正好解决了道路边界任务的后半段问题：

```text
BEV feature -> Vector Boundary
```

相比 BEV segmentation 再后处理，MapTRv2 有几个适合本项目的特点：

1. 直接输出 polyline，和道路边界目标形式一致；
2. 使用 query 表达地图元素，天然支持多条边界实例；
3. 通过 Hungarian matching 学习预测线和 GT 线之间的对应关系；
4. permutation-equivalent 建模可以缓解 polyline 点序不唯一的问题；
5. MapTRv2 的 one-to-many matching / dense supervision 有助于训练收敛。

这些机制的价值不是为了讲论文细节，而是为了说明：

```text
MapTRv2 是一个值得下注的 Vector Map 框架。
```

## 7. 已完成验证

### 7.1 验证一：官方 Image Baseline

目的：

```text
确认官方 MapTRv2 工程链路可以正常运行。
```

链路：

```text
multi-view image
  -> image BEV
  -> MapTRv2Head
  -> vector map
```

结果：

- 数据预处理完成；
- 训练链路跑通；
- eval 链路跑通；
- visualization 链路跑通。

结论：

```text
官方 MapTRv2 baseline 可以作为后续改造基线。
```

### 7.2 验证二：Image + LiDAR Fusion

目的：

```text
验证点云 BEV 是否可以接入 MapTRv2Head。
```

链路：

```text
Image BEV + LiDAR SparseEncoder BEV
  -> Fusion
  -> MapTRv2Head
```

结果：

- Image + LiDAR fusion 可以正常训练；
- 可以完成 eval；
- 可以完成 visualization。

结论：

```text
LiDAR 点云可以被编码成 BEV feature，并接入 MapTRv2Head。
```

### 7.3 验证三：Pure LiDAR Boundary-only

目的：

```text
验证点云是否可以直接驱动 MapTRv2Head 学习道路边界。
```

链路：

```text
AV2 LiDAR points
  -> voxelization
  -> SparseEncoder
  -> BEV feature
  -> MapTRv2Head
  -> boundary
```

任务收窄为：

```python
map_classes = ['boundary']
```

结果：

- pure LiDAR boundary-only 训练链路跑通；
- eval / visualization 跑通；
- 训练集可视化中出现接近 GT 的道路边界结构。

结论：

```text
点云 -> BEV -> Vector Boundary
这条子链路成立。
```

当前 val 上效果仍较弱，说明目前主要验证的是小样本过拟合能力，还没有证明跨场景泛化能力。

## 8. 当前风险项

目前验证的是：

```text
LiDAR -> Boundary
```

最终目标是：

```text
Radar -> Boundary
```

两者之间仍存在关键风险。

### 8.1 毫米波点云稀疏度

毫米波点数远少于 LiDAR，需要验证：

```text
道路边界在毫米波点云中是否足够可观测。
```

如果边界相关回波过少，单帧输入可能不足以支撑 vector boundary 学习。

### 8.2 道路边界回波稳定性

需要确认以下对象是否会产生稳定毫米波回波：

- 路缘石；
- 护栏；
- 绿化带边界；
- 道路边缘附近静态结构。

如果回波主要来自动车辆或零散强反射点，则需要额外做动静态过滤或多帧统计。

### 8.3 多帧累积依赖程度

需要比较：

```text
单帧 Radar
vs
多帧 Radar Accumulation
```

毫米波道路边界可能依赖多帧累积才能形成稳定空间结构。因此 ego-motion compensation 和时间对齐会成为关键。

### 8.4 GT 对齐误差

Boundary GT 与 Radar 坐标系之间的标定、同步误差会直接影响训练。

需要提前确认：

- radar2ego 外参精度；
- 时间戳同步；
- ego pose 质量；
- HD map / 标注边界与传感器坐标的一致性。

### 8.5 重复预测和后处理

MapTRv2 使用固定 query 输出多条候选线。boundary-only 场景中可能出现多个 query 重复预测同一段边界。

后续需要考虑：

- score threshold；
- query 数量；
- one-to-many 设置；
- 简单线段 NMS 或 polyline merge。

## 9. 下一阶段计划

### 9.1 扩大 AV2 LiDAR 验证

当前计划下载：

```text
train: 8 logs
val:   2 logs
```

目的：

- 验证 pure LiDAR boundary-only 是否能从小样本过拟合走向跨 log 泛化；
- 观察 train / val loss 和可视化差异；
- 调整 query 数、one-to-many、score threshold 和后处理策略。

### 9.2 抽象 Sensor-to-BEV 接口

将模型逻辑整理为：

```text
points
  -> BEV feature
  -> MapTRv2Head
```

目标是让 LiDAR encoder / Radar encoder 可以替换，而下游 vector decoder 保持一致。

### 9.3 启动毫米波小样本实验

毫米波阶段优先做小样本 overfit：

```text
radar train samples
  -> boundary-only overfit
  -> train visualization
```

如果 train 上能拟合，再进行 val 泛化实验。

### 9.4 强化 Boundary-only 设置

重点关注：

- 只保留 `boundary` 单类；
- 降低重复预测 query；
- 调整 `bbox_coder.max_num`；
- 尝试线段去重或简单 NMS；
- 对比不同 score threshold 的可视化结果。

## 10. 讨论问题

后续组内可以重点讨论：

1. 毫米波点云是否需要多帧累积作为默认输入？
2. 动态点应删除、降权，还是作为速度特征保留？
3. Radar BEV encoder 用 voxel / pillar / raster 哪种更合适？
4. 道路边界 GT 如何和毫米波坐标严格对齐？
5. 是否需要针对 boundary 做专门的线段去重后处理？
6. 下一阶段投入应优先放在 Radar BEV Encoder，还是继续完善 Vector Decoder？

## 11. 一句话总结

本次预研不是证明“毫米波已经可以提取道路边界”，而是证明：

```text
MapTRv2 路线具备继续投入毫米波验证的价值。
```

当前已证明：

```text
LiDAR BEV 能驱动 MapTRv2 Decoder 学习道路边界。
```

因此，当前主要风险已经收敛到：

```text
毫米波特征表达能力
```

而不是：

```text
Vector Map 路线本身
```

建议继续投入。下一阶段聚焦：

```text
Radar BEV Encoder
```
