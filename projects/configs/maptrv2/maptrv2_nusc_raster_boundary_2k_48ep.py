_base_ = ['./maptrv2_nusc_raster_boundary_2k.py']

total_epochs = 48
runner = dict(type='EpochBasedRunner', max_epochs=48)

# Continue saving every 6 epochs, so resumed training from epoch_24 will write
# epoch_30/36/42/48 checkpoints.
checkpoint_config = dict(max_keep_ckpts=4, interval=6)
