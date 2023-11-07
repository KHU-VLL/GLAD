import torch
import torch.nn as nn
from mmcv.cnn import trunc_normal_init

from ..builder import HEADS
from ...core import mean_class_accuracy
from .base import BaseDAContrastiveHead, get_fc_block_by_channels
from .contrastiveda_tsm_head import RunningAverage


@HEADS.register_module()
class ContrastiveDATransformerHead(BaseDAContrastiveHead):
    def __init__(self,
                num_classes,
                in_channels,
                channels=[],
                num_features=512,
                loss_cls=dict(type='SemisupervisedContrastiveLoss', unsupervised=True, loss_ratio=.35, tau=5.),
                centroids=dict(p_centroid=''),  # p_centroid would be "PXX_train_closed.pkl"
                init_std=0.02,
                dropout_ratio=0.5,
                **kwargs):
        super().__init__(num_classes, in_channels, loss_cls, **kwargs)
        self.init_std = init_std
        self.num_features = num_features
        self.fc_contra = get_fc_block_by_channels(in_channels, num_features, channels, dropout_ratio)
        self._init_centroids(centroids)

    def _init_centroids(self, centroids={}):
        if centroids.get('p_centroid', None):
            self.with_given_centroids = True
            import pickle
            with open(centroids['p_centroid'], 'rb') as f:
                self.centroids = torch.from_numpy(pickle.load(f)).cuda()
            assert self.centroids.shape[0] == self.num_classes, f'Size mismatched: {(self.centroids.shape[0], self.num_classes)}'
        else:  # centroids are needed only for scoring
            self.with_given_centroids = False
            self.centroids = [RunningAverage() for _ in range(self.num_classes)]

    def init_weights(self):
        """Initiate the parameters from scratch."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                trunc_normal_init(m, std=self.init_std)

    def forward(self, x, domains=None):
        # [4N, in_channels]
        cls_score = self.fc_contra(x)
        # [4N, num_classes]
        return cls_score
