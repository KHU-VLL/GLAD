import copy
import os.path as osp

import numpy as np
import torch

from .base import BaseDataset
from .rawframe_dataset import RawframeDataset
from .builder import DATASETS


@DATASETS.register_module()
class ContrastiveRawframeDataset(RawframeDataset):
    def __init__(self,
                 ann_file,
                 pipeline,
                 data_prefix=None,
                 test_mode=False,
                 filename_tmpl='img_{:05}.jpg',
                 with_offset=False,
                 multi_class=False,
                 num_classes=None,
                 start_index=1,
                 modality='RGB',
                 sample_by_class=False,
                 power=0.,
                 dynamic_length=False):
        super().__init__(
            ann_file,
            pipeline,
            data_prefix,
            test_mode,
            filename_tmpl,
            with_offset,
            multi_class,
            num_classes,
            start_index,
            modality,
            sample_by_class=sample_by_class,
            power=power,
            dynamic_length=dynamic_length
        )

    def prepare_train_frames(self, idx):
        results_view1 = super().prepare_train_frames(idx)
        results_view2 = super().prepare_train_frames(idx)
        result = {}
        for (k1, v1), (k2, v2) in zip(results_view1.items(), results_view2.items()):
            assert k1 == k2
            # [2, N, T, C, H, W] for 3D tensors
            # [2, N, C, H, W] for 2D tensors
            # [2] for other scalar infos
            result[k1] = torch.stack([v1, v2])
        return result

    def prepare_test_frames(self, idx):
        results_view1 = super().prepare_test_frames(idx)
        results_view2 = super().prepare_test_frames(idx)
        result = {}
        for (k1, v1), (k2, v2) in zip(results_view1.items(), results_view2.items()):
            assert k1 == k2
            # [2, N, T, C, H, W] for 3D tensors
            # [2, N, C, H, W] for 2D tensors
            # [2] for other scalar infos
            result[k1] = torch.stack([v1, v2])
        return result
