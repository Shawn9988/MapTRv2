_base_ = ['./maptrv2_nusc_raster_boundary_2k.py']

# Diagnostic config for visualization only.
# It keeps the trained model settings unchanged, but makes vis_pred.py read
# train samples so we can distinguish underfitting from poor validation generalization.
data = dict(
    test=dict(
        ann_file='work_dirs/nuscenes_2k_radar_map_samples_train/maptr_raster_infos.pkl',
        max_samples=None))
