_base_ = ['./maptrv2_nusc_raster_boundary_overfit.py']

work_dir = './work_dirs/maptrv2_xd_raster_boundary'

map_classes = ['boundary']
num_map_classes = 1

# XD samples store s_local in [0, 100], but MapTR vector heads assume the
# patch center is the origin. build_xd_maptr_infos.py shifts x by -50 m.
point_cloud_range = [-50.0, -40.0, -10.0, 50.0, 40.0, 10.0]
bev_h_ = 200
bev_w_ = 100
fixed_ptsnum_per_gt_line = 20

# GT 现为手工完整标注，关闭 partial positive（该机制仅为容忍算法生成的缺失 GT）
partial_positive = False
partial_bg_weight = 0.05
bev_seg=True

aux_seg_cfg = dict(
    use_aux_seg=True,
    bev_seg=bev_seg,
    pv_seg=False,
    seg_classes=1,
    feat_down_sample=32,
    pv_thickness=1)


model = dict(
    modality='raster',
    raster_encoder=dict(
        in_channels=3,
        out_channels=256,
        channels=(32, 64, 128, 256)),
    pts_bbox_head=dict(
        bev_h=bev_h_,
        bev_w=bev_w_,
        num_classes=num_map_classes,
        partial_positive=partial_positive,
        partial_bg_weight=partial_bg_weight,
        aux_seg=aux_seg_cfg,
        bbox_coder=dict(
            pc_range=point_cloud_range,
            post_center_range=[-55, -45, -55, -45, 55, 45, 55, 45],
            max_num=20,
            num_classes=num_map_classes),
        positional_encoding=dict(
            row_num_embed=bev_h_,
            col_num_embed=bev_w_),
        transformer=dict(
            modality='raster',
            fuser=dict(
                type='ConvFuser',
                in_channels=[256],
                out_channels=256))),
    train_cfg=dict(
        pts=dict(
            point_cloud_range=point_cloud_range,
            assigner=dict(pc_range=point_cloud_range))))

train_pipeline = [
    dict(
        type='LoadRasterMapImage',
        color_type='color',
        invert=False,
        normalize=True),
	dict(type='XDMapGeometricAug', 
	     prob=0.5, 
		 max_shift_m=1.5,
         max_rotate_deg=8.0, 
		 scale_range=(0.9, 1.1)), #新增，数据增广
    dict(
        type='CustomCollect3D',
        keys=['raster_img'],
        meta_keys=('filename', 'raster_img_path', 'sample_idx',
                   'scene_token', 'frame_idx', 'prev_idx', 'next_idx',
                   'timestamp', 'can_bus', 'lidar2global', 'pc_range',
                   'source_npz_path',
                   'image_size', 'raw_xy', 'map_location', 'local_annotation',
                   'global_annotation'))
]

# test_pipeline = train_pipeline
test_pipeline = [
    dict(
        type='LoadRasterMapImage',
        color_type='color',
        invert=False,
        normalize=True),
    dict(
        type='CustomCollect3D',
        keys=['raster_img'],
        meta_keys=('filename', 'raster_img_path', 'sample_idx',
                   'scene_token', 'frame_idx', 'prev_idx', 'next_idx',
                   'timestamp', 'can_bus', 'lidar2global', 'pc_range',
                   'source_npz_path',
                   'image_size', 'raw_xy', 'map_location', 'local_annotation',
                   'global_annotation'))
]

dataset_type = 'CustomRasterMapDataset'
data_root = 'work_dirs/'

data = dict(
    samples_per_gpu=2,
    workers_per_gpu=0,
    train=dict(
		type='RepeatDataset',
        times=3,
		dataset=dict(
        type=dataset_type,
        ann_file='work_dirs/xd_maptr_samples_train/maptr_raster_infos.pkl',
        pipeline=train_pipeline,
        test_mode=False,
        bev_size=(bev_h_, bev_w_),
        pc_range=point_cloud_range,
        fixed_ptsnum_per_line=fixed_ptsnum_per_gt_line,
        map_classes=map_classes,
        aux_seg=aux_seg_cfg,
        max_samples=None)),
    val=dict(
        type=dataset_type,
        ann_file='work_dirs/xd_maptr_samples_val/maptr_raster_infos.pkl',
        pipeline=test_pipeline,
        test_mode=True,
        bev_size=(bev_h_, bev_w_),
        pc_range=point_cloud_range,
        fixed_ptsnum_per_line=fixed_ptsnum_per_gt_line,
        map_classes=map_classes,
        aux_seg=aux_seg_cfg,
        max_samples=None,
        samples_per_gpu=1),
    test=dict(
        type=dataset_type,
        ann_file='work_dirs/xd_maptr_samples_val/maptr_raster_infos.pkl',
        pipeline=test_pipeline,
        test_mode=True,
        bev_size=(bev_h_, bev_w_),
        pc_range=point_cloud_range,
        fixed_ptsnum_per_line=fixed_ptsnum_per_gt_line,
        map_classes=map_classes,
        aux_seg=aux_seg_cfg,
        max_samples=None,
        samples_per_gpu=1))

optimizer = dict(lr=2e-4)
total_epochs = 90
runner = dict(type='EpochBasedRunner', max_epochs=total_epochs)
# Override inherited save_best from the nuScenes base config. The XD raster
# dataset returns XdMap_* metrics, so the checkpoint monitor key must match.
evaluation = dict(
    interval=10,
    metric='chamfer',
    save_best='XdMap_chamfer/mAP',
    rule='greater')
checkpoint_config = dict(max_keep_ckpts=5, interval=10)
log_config = dict(
    interval=10,
    hooks=[
        dict(type='TextLoggerHook')
    ])
default_hooks = dict(
    early_stopping=dict(
        type='EarlyStoppingHook',
        monitor='XdMap_chamfer/mAP', # 监控val矢量mAP，越高越好
        rule='max',            # max=指标提升才算变好；监控loss用min
        patience=15,           # 连续15epoch无提升则停止训练
        min_delta=0.0001,      # 提升小于该值视为波动，不算改善
        verbose=True           # 打印早停相关日志
    )
)
