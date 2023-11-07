from ..builder import HEADS

from .osbp_tsf_head import OSBPDINODAHead
from .dann_tsm_head import GradReverse

import torch
import torch.nn as nn
import numpy as np


@HEADS.register_module()
class DANNDINODAHead(OSBPDINODAHead):
    def __init__(
        self, in_channels, num_classes,
        loss_cls=dict(type='DANNLoss'),
        use_bn=False, norm_last_layer=True, nlayers=3, hidden_dim=2048, bottleneck_dim=256, **kwargs):

        if 'num_classes' not in loss_cls:
            loss_cls['num_classes'] = num_classes
        super().__init__(num_classes, in_channels, loss_cls, **kwargs)

        self.dino_cls = self._build_dino(in_channels, num_classes, use_bn, norm_last_layer, nlayers, hidden_dim, bottleneck_dim)
        self.dino_domain = self._build_dino(in_channels, 1, use_bn, norm_last_layer, nlayers, hidden_dim, bottleneck_dim)

    def forward(self, x, domains=None, gt_labels=None):
        is_training = domains is not None and domains.shape[0] > 0
        source_idx_mask = domains == 'source' if is_training else np.ones(x.shape[0], dtype=bool)
        source_idx_mask = torch.squeeze(torch.from_numpy(source_idx_mask))

        cls_score = self.dino_cls(x[source_idx_mask])
        if is_training:
            domain_score = self.dino_domain(GradReverse.apply(x))
            return [cls_score, domain_score]
        else:
            return cls_score

    def loss(self, cls_score:list, labels, domains, **kwargs):
        source_idx = torch.from_numpy(domains == 'source')
        labels = labels[source_idx]
        return super().loss(cls_score, labels, domains, **kwargs)

    @staticmethod
    def _build_dino(in_channels, out_dim, use_bn=False, norm_last_layer=True, nlayers=3, hidden_dim=2048, bottleneck_dim=256):
        nlayers = max(nlayers, 1)
        if nlayers == 1:
            mlp = nn.Linear(in_channels, bottleneck_dim)
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
            mlp = nn.Sequential(*layers)
        last_layer = nn.utils.weight_norm(nn.Linear(bottleneck_dim, out_dim, bias=False))
        last_layer.weight_g.data.fill_(1)
        if norm_last_layer:
            last_layer.weight_g.requires_grad = False
        return nn.Sequential(mlp, L2Norm(), last_layer)


class L2Norm(nn.Module):
    def forward(self, x):
        return nn.functional.normalize(x, dim=-1, p=2)
