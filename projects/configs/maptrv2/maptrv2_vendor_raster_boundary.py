_base_ = ['./maptrv2_nusc_raster_boundary_overfit.py']

map_classes = ['boundary']
num_map_classes = 1

# Vendor samples are exported in local Frenet coordinates:
# x = s_local, y = lateral d.
point_cloud_range = [0.0, -40.0, -10.0, 100.0, 40.0, 10.0]
bev_h_ = 200
bev_w_ = 100
fixed_ptsnum_per_gt_line = 20

aux_seg_cfg = dict(
    use_aux_seg=False,
    bev_seg=False,
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
        aux_seg=aux_seg_cfg,
        bbox_coder=dict(
            pc_range=point_cloud_range,
            post_center_range=[-5, -45, -5, -45, 105, 45, 105, 45],
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

test_pipeline = train_pipeline

dataset_type = 'CustomRasterMapDataset'
data_root = 'work_dirs/'

data = dict(
    samples_per_gpu=1,
    workers_per_gpu=0,
    train=dict(
        type=dataset_type,
        ann_file='work_dirs/vendor_maptr_samples/maptr_raster_infos.pkl',
        pipeline=train_pipeline,
        test_mode=False,
        bev_size=(bev_h_, bev_w_),
        pc_range=point_cloud_range,
        fixed_ptsnum_per_line=fixed_ptsnum_per_gt_line,
        map_classes=map_classes,
        aux_seg=aux_seg_cfg,
        max_samples=None),
    val=dict(
        type=dataset_type,
        ann_file='work_dirs/vendor_maptr_samples/maptr_raster_infos.pkl',
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
        ann_file='work_dirs/vendor_maptr_samples/maptr_raster_infos.pkl',
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
total_epochs = 24
runner = dict(type='EpochBasedRunner', max_epochs=24)
evaluation = dict(interval=1000)
checkpoint_config = dict(max_keep_ckpts=4, interval=6)
