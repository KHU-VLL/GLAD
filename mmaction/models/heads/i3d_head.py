# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
from mmcv.cnn import normal_init

from einops import repeat

from ..builder import HEADS
from .base import BaseHead


@HEADS.register_module()
class I3DHead(BaseHead):
    """Classification head for I3D.

    Args:
        num_classes (int): Number of classes to be classified.
        in_channels (int): Number of channels in input feature.
        loss_cls (dict): Config for building loss.
            Default: dict(type='CrossEntropyLoss')
        spatial_type (str): Pooling type in spatial dimension. Default: 'avg'.
        dropout_ratio (float): Probability of dropout layer. Default: 0.5.
        init_std (float): Std value for Initiation. Default: 0.01.
        kwargs (dict, optional): Any keyword argument to be used to initialize
            the head.
    """

    def __init__(self,
                 num_classes,
                 in_channels,
                 loss_cls=dict(type='CrossEntropyLoss'),
                 spatial_type='avg',
                 dropout_ratio=0.5,
                 init_std=0.01,
                 **kwargs):
        super().__init__(num_classes, in_channels, loss_cls, **kwargs)

        self.spatial_type = spatial_type
        self.dropout_ratio = dropout_ratio
        self.init_std = init_std
        if self.dropout_ratio != 0:
            self.dropout = nn.Dropout(p=self.dropout_ratio)
        else:
            self.dropout = None
        self.fc_cls = nn.Linear(self.in_channels, self.num_classes)

        if self.spatial_type == 'avg':
            # use `nn.AdaptiveAvgPool3d` to adaptively match the in_channels.
            self.avg_pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        else:
            self.avg_pool = None

    def init_weights(self):
        """Initiate the parameters from scratch."""
        normal_init(self.fc_cls, std=self.init_std)

    def forward(self, x, **kwargs):
        """Defines the computation performed at every call.

        Args:
            x (torch.Tensor): The input data.

        Returns:
            torch.Tensor: The classification scores for input samples.
        """
        # [N, in_channels, 4, 7, 7]
        if self.avg_pool is not None:
            x = self.avg_pool(x)
        # [N, in_channels, 1, 1, 1]
        if self.dropout is not None:
            x = self.dropout(x)
        # [N, in_channels, 1, 1, 1]
        x = x.view(x.shape[0], -1)
        # [N, in_channels]
        cls_score = self.fc_cls(x)
        # [N, num_classes]
        return cls_score

    def loss(self, cls_score, labels, domains=None, train=False, **kwargs):
        if cls_score.shape[0] > labels.shape[0]:  # multi-clip
            assert cls_score.shape[0] % labels.shape[0] == 0
            N = cls_score.shape[0] // labels.shape[0]
            labels = repeat(labels, 'b -> (b n)', n=N)

        losses = {}
        if domains is not None:
            source_idx = domains == 'source'
            target_idx = ~source_idx
            mca_source = self.calc_mca(cls_score[source_idx].detach(), labels[source_idx])
            mca_target = self.calc_mca(cls_score[target_idx], labels[target_idx])
            losses['s_mca'] = mca_source
            losses['t_mca'] = mca_target
        else:
            mca = self.calc_mca(cls_score, labels)
            losses = {'mca': mca}

        loss_cls = self.loss_cls(cls_score, labels, domains=domains, **kwargs)
        if isinstance(loss_cls, dict):
            losses.update(loss_cls)
        else:
            losses['loss_cls'] = loss_cls
        return losses
