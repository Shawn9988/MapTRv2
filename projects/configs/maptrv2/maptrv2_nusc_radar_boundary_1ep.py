_base_ = ['./maptrv2_nusc_lidar_boundary_1ep.py']

load_dim = 6
use_dim = [0, 1, 2, 3, 4, 5]
map_classes = ['boundary']

input_modality = dict(
    use_lidar=False,
    use_camera=False,
    use_radar=True,
    use_map=False,
    use_external=True)

model = dict(
    lidar_encoder=dict(
        backbone=dict(in_channels=len(use_dim))))

train_pipeline = [
    dict(
        type='CustomLoadRadarPointsFromMultiSweeps',
        coord_type='LIDAR',
        sweeps_num=3,
        load_dim=load_dim,
        use_dim=use_dim,
        test_mode=False,
        disable_filters=True),
    dict(
        type='DefaultFormatBundle3D',
        with_gt=False,
        with_label=False,
        class_names=map_classes),
    dict(type='CustomCollect3D', keys=['points'])
]

test_pipeline = [
    dict(
        type='CustomLoadRadarPointsFromMultiSweeps',
        coord_type='LIDAR',
        sweeps_num=3,
        load_dim=load_dim,
        use_dim=use_dim,
        test_mode=True,
        disable_filters=True),
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
    train=dict(
        pipeline=train_pipeline,
        modality=input_modality),
    val=dict(
        pipeline=test_pipeline,
        modality=input_modality),
    test=dict(
        pipeline=test_pipeline,
        modality=input_modality))
