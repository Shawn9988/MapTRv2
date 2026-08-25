_base_ = ['./maptrv2_nusc_r50_mini_1ep.py']

map_classes = ['boundary']
num_map_classes = 1

point_cloud_range = [-15.0, -30.0, -10.0, 15.0, 30.0, 10.0]
lidar_voxel_size = [0.2, 0.2, 0.5]
load_dim = 5
use_dim = [0, 1, 2, 3, 4]

aux_seg_cfg = dict(
    use_aux_seg=False,
    bev_seg=False,
    pv_seg=False,
    seg_classes=1,
    feat_down_sample=32,
    pv_thickness=1)

input_modality = dict(
    use_lidar=True,
    use_camera=False,
    use_radar=False,
    use_map=False,
    use_external=True)

model = dict(
    use_grid_mask=False,
    pretrained=None,
    img_backbone=None,
    img_neck=None,
    modality='lidar',
    lidar_encoder=dict(
        voxelize=dict(
            max_num_points=10,
            point_cloud_range=point_cloud_range,
            voxel_size=lidar_voxel_size,
            max_voxels=[60000, 80000]),
        backbone=dict(
            type='SparseEncoder',
            in_channels=len(use_dim),
            sparse_shape=[150, 300, 41],
            output_channels=256,
            order=('conv', 'norm', 'act'),
            encoder_channels=((16, 16, 32), (32, 32, 64), (64, 64, 128),
                              (128, 128)),
            encoder_paddings=([0, 0, 1], [0, 0, 1], [0, 0, [1, 1, 0]],
                              [0, 0]),
            block_type='basicblock')),
    pts_bbox_head=dict(
        num_classes=num_map_classes,
        aux_seg=aux_seg_cfg,
        bbox_coder=dict(
            max_num=20,
            num_classes=num_map_classes),
        transformer=dict(
            modality='lidar',
            fuser=dict(
                type='ConvFuser',
                in_channels=[512],
                out_channels=256))))

train_pipeline = [
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=load_dim,
        use_dim=use_dim),
    dict(
        type='DefaultFormatBundle3D',
        with_gt=False,
        with_label=False,
        class_names=map_classes),
    dict(type='CustomCollect3D', keys=['points'])
]

test_pipeline = [
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=load_dim,
        use_dim=use_dim),
    dict(
        type='MultiScaleFlipAug3D',
        img_scale=(1600, 900),
        pts_scale_ratio=1,
        flip=False,
        transforms=[
            dict(
                type='DefaultFormatBundle3D',
                with_gt=False,
                with_label=False,
                class_names=map_classes),
            dict(type='CustomCollect3D', keys=['points'])
        ])
]

data = dict(
    samples_per_gpu=1,
    workers_per_gpu=0,
    train=dict(
        pipeline=train_pipeline,
        map_classes=map_classes,
        modality=input_modality,
        aux_seg=aux_seg_cfg),
    val=dict(
        pipeline=test_pipeline,
        map_classes=map_classes,
        modality=input_modality,
        samples_per_gpu=1),
    test=dict(
        pipeline=test_pipeline,
        map_classes=map_classes,
        modality=input_modality,
        samples_per_gpu=1))

optimizer = dict(lr=2e-4)
total_epochs = 1
runner = dict(type='EpochBasedRunner', max_epochs=1)
evaluation = dict(interval=1)
checkpoint_config = dict(max_keep_ckpts=1, interval=1)
