import copy
import json
import os
import os.path as osp
import tempfile

import mmcv
import numpy as np
import torch
from mmcv.parallel import DataContainer as DC
from mmdet.datasets import DATASETS
from mmdet.datasets.pipelines import Compose, to_tensor
from torch.utils.data import Dataset

from .nuscenes_offlinemap_dataset import (LiDARInstanceLines,
                                          VectorizedLocalMap, output_to_vecs)


@DATASETS.register_module()
class CustomRasterMapDataset(Dataset):
    """Raster BEV input dataset with vector map GT."""

    CLASSES = ('map_element',)
    MAPCLASSES = ('boundary',)

    def __init__(self,
                 ann_file,
                 pipeline,
                 data_root=None,
                 test_mode=False,
                 bev_size=(200, 100),
                 pc_range=(-30.0, -40.0, -10.0, 30.0, 40.0, 10.0),
                 fixed_ptsnum_per_line=20,
                 eval_use_same_gt_sample_num_flag=True,
                 padding_value=-10000,
                 map_classes=None,
                 aux_seg=dict(
                     use_aux_seg=False,
                     bev_seg=False,
                     pv_seg=False,
                     seg_classes=1,
                     feat_down_sample=32),
                 max_samples=None,
                 **kwargs):
        self.ann_file = ann_file
        self.data_root = data_root
        self.test_mode = test_mode
        self.pipeline = Compose(pipeline)
        self.bev_size = bev_size
        self.pc_range = list(pc_range)
        self.MAPCLASSES = self.get_map_classes(map_classes)
        self.NUM_MAPCLASSES = len(self.MAPCLASSES)
        self.CLASSES = tuple(self.MAPCLASSES)
        self.cat2id = {name: i for i, name in enumerate(self.CLASSES)}
        self.fixed_ptsnum_per_line = fixed_ptsnum_per_line
        self.eval_use_same_gt_sample_num_flag = eval_use_same_gt_sample_num_flag
        self.padding_value = padding_value
        self.aux_seg = aux_seg
        self.max_samples = max_samples
        # eval needs: number of pred/gt points per instance and a GT cache file
        self.fixed_num = fixed_ptsnum_per_line
        self.modality = kwargs.get('modality', dict(
            use_lidar=False, use_camera=False, use_radar=False,
            use_map=False, use_external=True))
        # cache the formatted GT next to the pkl, so each split is self-contained
        self.map_ann_file = osp.join(osp.dirname(ann_file), 'maptr_map_gt.json')

        self.data_infos = self.load_annotations(ann_file)
        self.flag = np.zeros(len(self.data_infos), dtype=np.uint8)

        patch_h = self.pc_range[4] - self.pc_range[1]
        patch_w = self.pc_range[3] - self.pc_range[0]
        self.patch_size = (patch_h, patch_w)
        self.vector_map = VectorizedLocalMap(
            canvas_size=bev_size,
            patch_size=self.patch_size,
            map_classes=self.MAPCLASSES,
            fixed_ptsnum_per_line=fixed_ptsnum_per_line,
            padding_value=padding_value,
            aux_seg=aux_seg)

    @classmethod
    def get_map_classes(cls, map_classes=None):
        if map_classes is None:
            return cls.MAPCLASSES
        if isinstance(map_classes, str):
            return mmcv.list_from_file(map_classes)
        if isinstance(map_classes, (tuple, list)):
            return list(map_classes)
        raise ValueError(f'Unsupported map_classes type {type(map_classes)}')

    def load_annotations(self, ann_file):
        data = mmcv.load(ann_file)
        infos = data['infos']
        if self.max_samples is not None and self.max_samples > 0:
            infos = infos[:self.max_samples]
        self.metadata = data.get('metadata', {})
        return infos

    def __len__(self):
        return len(self.data_infos)

    def pre_pipeline(self, results):
        results['bbox3d_fields'] = []
        results['pts_mask_fields'] = []
        results['pts_seg_fields'] = []
        results['bbox_fields'] = []
        results['mask_fields'] = []
        results['seg_fields'] = []

    def get_data_info(self, index):
        info = self.data_infos[index]
        input_dict = dict(
            sample_idx=info['token'],
            raster_img_path=info['bev_img_path'],
            source_npz_path=info.get('source_npz_path', None),
            ann_info=info['annotation'],
            annotation=info['annotation'],
            scene_token=info.get('scene_token', ''),
            frame_idx=index,
            prev_idx='',
            next_idx='',
            timestamp=info.get('timestamp', index),
            can_bus=np.zeros(18, dtype=np.float32),
            lidar2global=np.asarray(
                info.get('lidar2global', np.eye(4)), dtype=np.float32),
            map_location=info.get('map_location', None),
            pc_range=info.get('pc_range', self.pc_range),
            image_size=info.get('image_size', None),
            raw_xy=info.get('raw_xy', self.metadata.get('raw_xy', False)),
            local_annotation=info.get(
                'raster_annotation', info.get('annotation', {})),
            global_annotation=info.get(
                'global_raster_annotation',
                info.get('global_annotation', {})),
        )
        return input_dict

    def vectormap_pipeline(self, example, input_dict):
        anns_results = self.vector_map.gen_vectorized_samples(
            input_dict['annotation'],
            example=example,
            feat_down_sample=self.aux_seg['feat_down_sample'])

        gt_vecs_label = to_tensor(anns_results['gt_vecs_label'])
        if isinstance(anns_results['gt_vecs_pts_loc'], LiDARInstanceLines):
            gt_vecs_pts_loc = anns_results['gt_vecs_pts_loc']
        else:
            gt_vecs_pts_loc = to_tensor(anns_results['gt_vecs_pts_loc'])
            try:
                gt_vecs_pts_loc = gt_vecs_pts_loc.flatten(1).to(
                    dtype=torch.float32)
            except Exception:
                pass
        example['gt_labels_3d'] = DC(gt_vecs_label, cpu_only=False)
        example['gt_bboxes_3d'] = DC(gt_vecs_pts_loc, cpu_only=True)
        if anns_results['gt_semantic_mask'] is not None:
            example['gt_seg_mask'] = DC(
                to_tensor(anns_results['gt_semantic_mask']), cpu_only=False)
        if anns_results['gt_pv_semantic_mask'] is not None:
            example['gt_pv_seg_mask'] = DC(
                to_tensor(anns_results['gt_pv_semantic_mask']), cpu_only=False)
        return example

    def prepare_train_data(self, index):
        input_dict = self.get_data_info(index)
        self.pre_pipeline(input_dict)
        example = self.pipeline(input_dict)
        example = self.vectormap_pipeline(example, input_dict)
        return example

    def prepare_test_data(self, index):
        input_dict = self.get_data_info(index)
        self.pre_pipeline(input_dict)
        example = self.pipeline(input_dict)
        example = self.vectormap_pipeline(example, input_dict)
        return example

    def __getitem__(self, idx):
        if self.test_mode:
            return self.prepare_test_data(idx)
        return self.prepare_train_data(idx)

    def _format_gt(self):
        """Convert GT vectors to the json format expected by eval_map.

        Always regenerates (overwrites) so it never goes stale after the pkl
        is rebuilt.
        """
        gt_annos = []
        print('Start to convert gt map format...')
        prog_bar = mmcv.ProgressBar(len(self))
        for sample_id in range(len(self)):
            gt_sample = self.vectormap_pipeline({}, self.data_infos[sample_id])
            gt_labels = gt_sample['gt_labels_3d'].data.numpy()
            gt_vecs = gt_sample['gt_bboxes_3d'].data.instance_list
            gt_vec_list = []
            for gt_label, gt_vec in zip(gt_labels, gt_vecs):
                pts = np.array(list(gt_vec.coords))
                gt_vec_list.append(dict(
                    pts=pts,
                    pts_num=len(pts),
                    cls_name=self.MAPCLASSES[gt_label],
                    type=int(gt_label)))
            gt_annos.append(dict(
                sample_token=self.data_infos[sample_id]['token'],
                vectors=gt_vec_list))
            prog_bar.update()
        mmcv.mkdir_or_exist(osp.dirname(self.map_ann_file))
        mmcv.dump({'GTs': gt_annos}, self.map_ann_file)
        print('\nGT anns written to', self.map_ann_file)

    def _format_bbox(self, results, jsonfile_prefix):
        pred_annos = []
        print('Start to convert map detection format...')
        for sample_id, det in enumerate(mmcv.track_iter_progress(results)):
            if isinstance(det, dict) and 'pts_bbox' in det:
                det = det['pts_bbox']  # 模型输出外层包了一层
            vecs = output_to_vecs(det)
            pred_vec_list = []
            for vec in vecs:
                pred_vec_list.append(dict(
                    pts=vec['pts'],
                    pts_num=len(vec['pts']),
                    cls_name=self.MAPCLASSES[vec['label']],
                    type=int(vec['label']),
                    confidence_level=vec['score']))
            pred_annos.append(dict(
                sample_token=self.data_infos[sample_id]['token'],
                vectors=pred_vec_list))

        self._format_gt()
        mmcv.mkdir_or_exist(jsonfile_prefix)
        res_path = osp.join(jsonfile_prefix, 'xd_map_results.json')
        print('Results written to', res_path)
        mmcv.dump({'meta': self.modality, 'results': pred_annos}, res_path)
        return res_path

    def format_results(self, results, jsonfile_prefix=None):
        if jsonfile_prefix is None:
            tmp_dir = tempfile.TemporaryDirectory()
            jsonfile_prefix = osp.join(tmp_dir.name, 'results')
        else:
            tmp_dir = None
        result_files = self._format_bbox(results, jsonfile_prefix)
        return result_files, tmp_dir

    def _evaluate_single(self, result_path, logger=None, metric='chamfer'):
        from projects.mmdet3d_plugin.datasets.map_utils.mean_ap import (
            eval_map, format_res_gt_by_classes)
        result_path = osp.abspath(result_path)
        detail = dict()

        with open(result_path, 'r') as f:
            gen_results = json.load(f)['results']
        with open(self.map_ann_file, 'r') as f:
            annotations = json.load(f)['GTs']

        cls_gens, cls_gts = format_res_gt_by_classes(
            result_path, gen_results, annotations,
            cls_names=self.MAPCLASSES,
            num_pred_pts_per_instance=self.fixed_num,
            eval_use_same_gt_sample_num_flag=self.eval_use_same_gt_sample_num_flag,
            pc_range=self.pc_range)

        metrics = metric if isinstance(metric, list) else [metric]
        for metric in metrics:
            if metric not in ('chamfer', 'iou'):
                raise KeyError(f'metric {metric} is not supported')
            if metric == 'chamfer':
                thresholds = [0.5, 1.0, 1.5]
            else:
                thresholds = np.linspace(
                    .5, 0.95, int(np.round((0.95 - .5) / .05)) + 1,
                    endpoint=True)
            cls_aps = np.zeros((len(thresholds), self.NUM_MAPCLASSES))
            for i, thr in enumerate(thresholds):
                print('-*' * 10 + f'threshold:{thr}' + '-*' * 10)
                _, cls_ap = eval_map(
                    gen_results, annotations, cls_gens, cls_gts,
                    threshold=thr, cls_names=self.MAPCLASSES, logger=logger,
                    num_pred_pts_per_instance=self.fixed_num,
                    pc_range=self.pc_range, metric=metric)
                for j in range(self.NUM_MAPCLASSES):
                    cls_aps[i, j] = cls_ap[j]['ap']
            for i, name in enumerate(self.MAPCLASSES):
                print('{}: {}'.format(name, cls_aps.mean(0)[i]))
                detail['XdMap_{}/{}_AP'.format(metric, name)] = cls_aps.mean(0)[i]
            print('map: {}'.format(cls_aps.mean(0).mean()))
            detail['XdMap_{}/mAP'.format(metric)] = cls_aps.mean(0).mean()
            for i, name in enumerate(self.MAPCLASSES):
                for j, thr in enumerate(thresholds):
                    detail['XdMap_{}/{}_AP_thr_{}'.format(metric, name, thr)] = cls_aps[j][i]
        return detail

    def evaluate(self, results, metric='chamfer', logger=None,
                 jsonfile_prefix=None, **kwargs):
        result_files, tmp_dir = self.format_results(results, jsonfile_prefix)
        results_dict = self._evaluate_single(result_files, metric=metric)
        if tmp_dir is not None:
            tmp_dir.cleanup()
        return results_dict
