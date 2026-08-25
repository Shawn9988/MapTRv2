_base_ = ['./maptrv2_nusc_r50_24ep.py']

model = dict(
    pretrained=dict(img='ckpts/resnet50-19c8e357.pth'),
)

data = dict(
    samples_per_gpu=1,
    workers_per_gpu=0,
    val=dict(samples_per_gpu=1),
    test=dict(samples_per_gpu=1),
)

total_epochs = 1
runner = dict(type='EpochBasedRunner', max_epochs=1)

evaluation = dict(interval=1)
checkpoint_config = dict(max_keep_ckpts=1, interval=1)

log_config = dict(
    interval=10,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook')
    ])
