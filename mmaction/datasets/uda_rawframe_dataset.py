import copy
import os.path as osp

import numpy as np
import torch

from .base import BaseDataset
from .rawframe_dataset import RawframeDataset
from .builder import DATASETS


@DATASETS.register_module()
class UDARawframeDataset(RawframeDataset):
    """RawframeDataset for DA. Referenced RawframeDataset"""
    def __init__(self,
                 source_ann_file,
                 target_ann_file,
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
                 power=None,
                 dynamic_length=False):
        self.filename_tmpl = filename_tmpl
        self.with_offset = with_offset
        self.source_ann_file = source_ann_file
        self.target_ann_file = target_ann_file
        super().__init__(
            ann_file=source_ann_file,  # actually not used
            pipeline=pipeline,
            data_prefix=data_prefix,
            test_mode=test_mode,
            filename_tmpl=filename_tmpl,
            with_offset=with_offset,
            multi_class=multi_class,
            num_classes=num_classes,
            start_index=start_index,
            modality=modality,
            sample_by_class=sample_by_class,
            power=power,
            dynamic_length=dynamic_length
        )

    def load_annotations(self):
        """Load annotation file to get video(a clip, actually) information
        
        Returns: list of dicts, list of metadata of each clip
            keys of video_info
                - frame_dir (str): The relative path to frame_dir.
                - offset (int): The index of the start frame of this clip. Set only if self.with_offset is True
                - label (int ot list of ints): The label of this clip, and is a list only if self.multi_class is True.
        """
        assert not self.source_ann_file.endswith('.json')
        assert not self.target_ann_file.endswith('.json')
        video_infos = []
        with open(self.source_ann_file, 'r') as f_source, open(self.target_ann_file, 'r') as f_target:
            video_infos += [self.extract_video_info(line, 'source') for line in f_source]
            video_infos += [self.extract_video_info(line, 'target') for line in f_target]
        return video_infos

    def extract_video_info(self, line: str, domain: str='source'):
        line_split = line.strip().split()
        video_info = {'domain': 0 if domain=='source' else 1}
        idx = 0
        # idx for frame_dir
        frame_dir = line_split[idx]
        if self.data_prefix is not None:
            frame_dir = osp.join(self.data_prefix, frame_dir)
        video_info['frame_dir'] = frame_dir
        idx += 1
        if self.with_offset:
            # idx for offset and total_frames
            video_info['offset'] = int(line_split[idx])
            video_info['total_frames'] = int(line_split[idx + 1])
            idx += 2
        else:
            # idx for total_frames
            video_info['total_frames'] = int(line_split[idx])
            idx += 1
        # idx for label[s]
        label = [int(x) for x in line_split[idx:]]
        assert label, f'missing label in line: {line}'
        if self.multi_class:
            assert self.num_classes is not None
            video_info['label'] = label
        else:
            assert len(label) == 1
            video_info['label'] = label[0]
        return video_info
