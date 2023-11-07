import torch

from .video_dataset import VideoDataset
from .builder import DATASETS


@DATASETS.register_module()
class ContrastiveVideoDataset(VideoDataset):
    def __init__(self, ann_file, pipeline, start_index=0, **kwargs):
        super().__init__(ann_file, pipeline, start_index=start_index, **kwargs)

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
