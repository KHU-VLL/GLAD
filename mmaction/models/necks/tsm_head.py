import numpy as np
import torch
import torch.nn as nn

from einops import rearrange, reduce

from mmcv.cnn import normal_init
from ..builder import NECKS, build_loss


@NECKS.register_module()
class TemporalLocalityAwareTSMNeck(nn.Module):
    def __init__(self,
        num_classes,
        in_channels,
        num_segments=8,
        loss_cls=dict(type='CrossEntropyLoss'),
        dropout_ratio=0.5,
        init_std=0.001,
        openset=False,
        **kwargs
    ):
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.num_segments = num_segments
        self.dropout_ratio = dropout_ratio
        self.init_std = init_std
        self.openset = openset

        self.loss_cls = build_loss(loss_cls)
        self.fc_cls = nn.Linear(self.in_channels, self.num_classes)

    def init_weights(self):
        normal_init(self.fc_cls, std=self.init_std)

    def forward(self,
        f_tallies,
        labels=None, domains=None,
        train=False,
        **kwargs
    ):
        pass
