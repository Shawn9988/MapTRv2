_base_ = ['./maptrv2_nusc_radar_boundary_1ep.py']

overfit_samples = 16
overfit_epochs = 60

load_dim = 6
use_dim = [0, 1, 2, 3, 4, 5]
map_classes = ['boundary']
data_root = 'data/nuscenes/'

train_pipeline = [
    dict(
        type='CustomLoadRadarPointsFromMultiSweeps',
        coord_type='LIDAR',
        sweeps_num=5,
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
        sweeps_num=5,
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
    samples_per_gpu=1,
    workers_per_gpu=0,
    train=dict(
        pipeline=train_pipeline,
        max_samples=overfit_samples),
    val=dict(
        ann_file=data_root + 'nuscenes_map_infos_temporal_train.pkl',
        map_ann_file=data_root + 'nuscenes_map_anns_train_radar_overfit.json',
        pipeline=test_pipeline,
        max_samples=overfit_samples,
        samples_per_gpu=1),
    test=dict(
        ann_file=data_root + 'nuscenes_map_infos_temporal_train.pkl',
        map_ann_file=data_root + 'nuscenes_map_anns_train_radar_overfit.json',
        pipeline=test_pipeline,
        max_samples=overfit_samples,
        samples_per_gpu=1))

optimizer = dict(lr=5e-4)
total_epochs = overfit_epochs
runner = dict(type='EpochBasedRunner', max_epochs=overfit_epochs)
evaluation = dict(interval=10)
checkpoint_config = dict(max_keep_ckpts=3, interval=10)
