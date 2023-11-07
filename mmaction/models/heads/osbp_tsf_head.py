import torch
import torch.nn as nn
from torch.nn.init import trunc_normal_

from ..builder import HEADS
from .base import BaseDAHead
from .osbp_tsm_head import GradReverse

# TODO: Make MLP-layered
@HEADS.register_module()
class OSBPDINODAHead(BaseDAHead):
    def __init__(
        self, in_channels, num_classes,
        loss_cls=dict(type='OSBPLoss'),
        use_bn=False, norm_last_layer=True, nlayers=3, hidden_dim=2048, bottleneck_dim=256, **kwargs):

        if 'num_classes' not in loss_cls:
            loss_cls['num_classes'] = num_classes
        super().__init__(num_classes, in_channels, loss_cls, **kwargs)  # call grandparent method

        nlayers = max(nlayers, 1)
        if nlayers == 1:
            self.mlp = nn.Linear(in_channels, bottleneck_dim)
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
        self.last_layer = nn.utils.weight_norm(nn.Linear(bottleneck_dim, num_classes, bias=False))
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

    def forward(self, x, num_segs=None, domains=None, gt_labels=None):
        if domains is not None and domains.shape[0] > 0:  # if train
            target_idx_mask = torch.squeeze(torch.from_numpy(domains == 'target'))
            target_idx_mask = target_idx_mask.repeat(self.num_segments)
            x = GradReverse.apply(x, target_idx_mask)
        # [2B*2, in_channels]
        x = self.mlp(x)
        x = nn.functional.normalize(x, dim=-1, p=2)
        x = self.last_layer(x)  # [2B*2, num_classes]
        return x


@HEADS.register_module()
class OSBPSVTHead(BaseDAHead):
    pass
