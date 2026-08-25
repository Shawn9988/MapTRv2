_base_ = ['./maptrv2_av2_3d_r50_lidar_fusion_1ep.py']

map_classes = ['boundary']
num_map_classes = 1
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
        type='CustomLoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=4,
        use_dim=[0, 1, 2, 3]),
    dict(
        type='DefaultFormatBundle3D',
        with_gt=False,
        with_label=False,
        class_names=map_classes),
    dict(type='CustomCollect3D', keys=['points'])
]

test_pipeline = [
    dict(
        type='CustomLoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=4,
        use_dim=[0, 1, 2, 3]),
    dict(
        type='MultiScaleFlipAug3D',
        img_scale=(2048, 2048),
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
        modality=input_modality))

optimizer = dict(lr=2e-4)
total_epochs = 1
runner = dict(max_epochs=1)
