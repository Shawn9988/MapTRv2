_base_ = ['./maptrv2_av2_3d_r50_6ep.py']

point_cloud_range = [-30.0, -15.0, -5.0, 30.0, 15.0, 3.0]
lidar_voxel_size = [0.2, 0.2, 0.2]

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    to_rgb=True)

input_modality = dict(
    use_lidar=True,
    use_camera=True,
    use_radar=False,
    use_map=False,
    use_external=True)

load_dim = 4
use_dim = [0, 1, 2, 3]

model = dict(
    modality='fusion',
    lidar_encoder=dict(
        voxelize=dict(
            max_num_points=10,
            point_cloud_range=point_cloud_range,
            voxel_size=lidar_voxel_size,
            max_voxels=[60000, 80000]),
        backbone=dict(
            type='SparseEncoder',
            in_channels=len(use_dim),
            sparse_shape=[300, 150, 41],
            output_channels=256,
            order=('conv', 'norm', 'act'),
            encoder_channels=((16, 16, 32), (32, 32, 64), (64, 64, 128),
                              (128, 128)),
            encoder_paddings=([0, 0, 1], [0, 0, 1], [0, 0, [1, 1, 0]],
                              [0, 0]),
            block_type='basicblock')),
    pts_bbox_head=dict(
        num_vec_one2one=20,
        num_vec_one2many=60,
        k_one2many=3,
        transformer=dict(
            modality='fusion',
            fuser=dict(
                type='ConvFuser',
                in_channels=[256, 512],
                out_channels=256),
            decoder=dict(
                transformerlayers=dict(
                    num_vec=20)))))

train_pipeline = [
    dict(type='CustomLoadMultiViewImageFromFiles', to_float32=True, padding=True),
    dict(
        type='CustomLoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=load_dim,
        use_dim=use_dim),
    dict(type='RandomScaleImageMultiViewImage', scales=[0.3]),
    dict(type='PhotoMetricDistortionMultiViewImage'),
    dict(type='NormalizeMultiviewImage', **img_norm_cfg),
    dict(type='PadMultiViewImage', size_divisor=32),
    dict(
        type='DefaultFormatBundle3D',
        with_gt=False,
        with_label=False,
        class_names=['divider', 'ped_crossing', 'boundary']),
    dict(type='CustomCollect3D', keys=['img', 'points'])
]

test_pipeline = [
    dict(type='CustomLoadMultiViewImageFromFiles', to_float32=True, padding=True),
    dict(
        type='CustomLoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=load_dim,
        use_dim=use_dim),
    dict(type='RandomScaleImageMultiViewImage', scales=[0.3]),
    dict(type='NormalizeMultiviewImage', **img_norm_cfg),
    dict(
        type='MultiScaleFlipAug3D',
        img_scale=(2048, 2048),
        pts_scale_ratio=1,
        flip=False,
        transforms=[
            dict(type='PadMultiViewImage', size_divisor=32),
            dict(
                type='DefaultFormatBundle3D',
                with_gt=False,
                with_label=False,
                class_names=['divider', 'ped_crossing', 'boundary']),
            dict(type='CustomCollect3D', keys=['img', 'points'])
        ])
]

data = dict(
    samples_per_gpu=1,
    workers_per_gpu=0,
    train=dict(
        pipeline=train_pipeline,
        modality=input_modality),
    val=dict(
        pipeline=test_pipeline,
        modality=input_modality,
        samples_per_gpu=1),
    test=dict(
        pipeline=test_pipeline,
        modality=input_modality))

optimizer = dict(lr=2e-4)
total_epochs = 1
runner = dict(max_epochs=1)
