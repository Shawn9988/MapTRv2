_base_ = ['./maptrv2_nusc_raster_boundary_overfit.py']

data = dict(
    train=dict(
        ann_file='work_dirs/nuscenes_2k_radar_map_samples_train/maptr_raster_infos.pkl',
        max_samples=None),
    val=dict(
        ann_file='work_dirs/nuscenes_2k_radar_map_samples_val/maptr_raster_infos.pkl',
        max_samples=None),
    test=dict(
        ann_file='work_dirs/nuscenes_2k_radar_map_samples_val/maptr_raster_infos.pkl',
        max_samples=None))

optimizer = dict(lr=2e-4)
total_epochs = 24
runner = dict(type='EpochBasedRunner', max_epochs=24)
evaluation = dict(interval=1000)
checkpoint_config = dict(max_keep_ckpts=4, interval=6)
