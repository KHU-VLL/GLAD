from .contrastiveda_transformer_head import ContrastiveDATransformerHead
from ..builder import HEADS
from .base import BaseHead

import torch
import torch.nn as nn
from torch.nn.init import trunc_normal_


class BaseDINOHead:
    def __init__(
        self, in_channels, out_dim,
        use_bn=False, norm_last_layer=True, nlayers=3, hidden_dim=2048, bottleneck_dim=256, **kwargs):

        nlayers = max(nlayers, 1)
        if nlayers == 1:
            self.mlp = nn.Sequential(nn.Linear(in_channels, bottleneck_dim))
        else:
            layers = [nn.Linear(in_channels, hidden_dim)]
            if use_bn:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.GELU())
            for _ in range(nlayers - 2):
                layers.append(nn.Linear(hidden_dim, hidden_dim))
                if use_bn:
                    layers.append(nn.BatchNorm1d(hidden_dim))
                layers.append(nn.GELU())
            layers.append(nn.Linear(hidden_dim, bottleneck_dim))
            self.mlp = nn.Sequential(*layers)
        self.last_layer = nn.utils.weight_norm(nn.Linear(bottleneck_dim, out_dim, bias=False))
        self.last_layer.weight_g.data.fill_(1)
        if norm_last_layer:
            self.last_layer.weight_g.requires_grad = False

    def init_weights(self):
        def _init_weight(m):
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=.02)
                if isinstance(m, nn.Linear) and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        for m in self.mlp:
            _init_weight(m)
        _init_weight(self.last_layer)

    def forward(self, x):
        # [2B*2, in_channels]
        x = self.mlp(x)
        x = nn.functional.normalize(x, dim=-1, p=2)
        x = self.last_layer(x)  # [2B*2, out_dim]
        return x


@HEADS.register_module()
class DINOHead(
    BaseDINOHead,  # init_weights, forward
    BaseHead  # others
):
    def __init__(
        self, in_channels, num_classes,
        loss_cls=dict(type='CrossEntropyLoss'),
        use_bn=False, norm_last_layer=True, nlayers=3, hidden_dim=2048, bottleneck_dim=256, **kwargs):
            BaseHead.__init__(self, num_classes, in_channels, loss_cls, **kwargs)
            BaseDINOHead.__init__(self, in_channels, num_classes, use_bn, norm_last_layer, nlayers, hidden_dim, bottleneck_dim, **kwargs)


@HEADS.register_module()
class DINODAHead(
    BaseDINOHead,  # init_weights, forward
    ContrastiveDATransformerHead  # others
):
    def __init__(
        self, in_channels, out_dim,
        loss_cls=dict(type='SemisupervisedContrastiveLoss'),
        use_bn=False, norm_last_layer=True, nlayers=3, hidden_dim=2048, bottleneck_dim=256, **kwargs):
            super(ContrastiveDATransformerHead, self).__init__(out_dim, in_channels, loss_cls, **kwargs)  # call grandparent(BaseDAContrastiveHead) method
            BaseDINOHead.__init__(self, in_channels, out_dim, use_bn, norm_last_layer, nlayers, hidden_dim, bottleneck_dim, **kwargs)
            self._init_centroids()

    def forward(self, x, domains=None, gt_labels=None):
        return super().forward(x)


@HEADS.register_module()
class TSMDINODAHead(DINODAHead):
    def __init__(self, num_segments=8, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_segments = num_segments
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x, num_segs=None, domains=None, gt_labels=None):
        # [2B*2*T, in_channels, 7, 7]
        x = self.avg_pool(x)  # [2B*2*T, in_channels, 1, 1]
        x = torch.flatten(x, start_dim=1)  # [2B*2*T, in_channels]
        x = x.reshape(-1, self.num_segments, self.in_channels)  # [2B*2, T, in_channels]
        x = super().forward(x)  # [2B*2, T, out_dim]
        x = x.mean(dim=1)  # [2B*2, out_dim]
        return x
