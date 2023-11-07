# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
from mmcv.cnn import trunc_normal_init
from einops import rearrange, reduce

from ..builder import HEADS
from .base import BaseHead


@HEADS.register_module()
class PlacesHead(BaseHead):
    def __init__(self,
                 num_classes,
                 in_channels,
                 batch_size,
                 loss_cls=dict(type='CrossEntropyLoss'),
                 init_std=0.02,
                 **kwargs):
        super().__init__(num_classes, in_channels, loss_cls, **kwargs)
        self.init_std = init_std
        self.batch_size = batch_size
        self.fc = nn.Linear(self.in_channels, self.num_classes)
        self.topk = 5

    def init_weights(self):
        """Initiate the parameters from scratch."""
        trunc_normal_init(self.fc, std=self.init_std)

    def forward(self, x, *args):
        # [N*T, C_in, 7, 7]
        x = nn.AdaptiveAvgPool2d(1)(x)
        # [N*T, C_in, 1, 1]
        x = torch.flatten(x, 1)
        # [N*T, C_in]
        cls_score = self.fc(x)
        # [N*T, K]
        if cls_score.shape[0] % self.batch_size == 0:
            cls_score = rearrange(cls_score, '(n t) k -> n t k', n=self.batch_size)
        else:  # last batch
            cls_score = rearrange(cls_score, '(n t) k -> n t k', t=30)
        # [N, T, K]
        _, indices = cls_score.sort(dim=-1, descending=True)  # [N, T, K], indices
        top_indices = indices[:,:,:self.topk]                 # [N, T, topK]
        one_hots = torch.eye(self.num_classes)[top_indices]   # [N, T, topK, K]
        cls_score = reduce(one_hots, 'n t topk k -> n k', 'sum')
        # [N, K], frequencies
        return cls_score
