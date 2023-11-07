from ..builder import HEADS
from .base import BaseHead

from mmcv.cnn import normal_init

import torch
import torch.nn as nn

import math
from itertools import combinations, permutations


@HEADS.register_module()
class TSMTOLHead(BaseHead):
    """Video temporal order prediction with PFE (Pairwire Feature Extraction), the same as OPN."""
    def __init__(self,
            num_clips,
            in_channels,
            num_hidden=512,
            num_segments=8,
            loss_cls=dict(type='CrossEntropyLoss'),
            spatial_type='avg',
            consensus=dict(type='AvgConsensus', dim=1),
            dropout_ratio=0.8,
            init_std=0.001,
            is_shift=True,
            as_neck=False,
            is_aux=False,
            **kwargs
        ):
        """
        Args:
            num_features (int): 512
        """
        # todo: parameter 맞춰줘야 됨
        self.num_clips = num_clips
        self.class_num = math.factorial(self.num_clips)
        super().__init__(self.class_num, in_channels, loss_cls, **kwargs)
        self.num_segments = num_segments
        self.num_hidden = num_hidden

        self.init_std = init_std
        self.is_shift = is_shift
        self.spatial_type = spatial_type

        self.avg_pool = nn.AdaptiveAvgPool2d(1) if self.spatial_type == 'avg' else nn.Identity()

        self.fc_cop1 = nn.Linear(self.in_channels*2, self.num_hidden)
        self.pair_num = int(self.num_clips*(self.num_clips-1)/2)  # NC2
        self.fc_cop2 = nn.Linear(self.num_hidden*self.pair_num, self.class_num)

        self.dropout = nn.Dropout(p=dropout_ratio)
        self.relu = nn.ReLU(inplace=True)

        self.as_neck = as_neck
        self.is_aux = is_aux

    def init_weights(self):
        """Initiate the parameters from scratch."""
        for layer in [self.fc_cop1, self.fc_cop2]:
            normal_init(layer, std=self.init_std)

    def forward(self, f, num_segs=None, domains=None, gt_labels=None):
        # B' = 2B or 2B * 2
        # f: [B'*N*T, C_feat, 7, 7]
        ff = self.avg_pool(f)  # [B'*N*T, C_feat, 1, 1]
        ff = torch.flatten(ff, start_dim=1)  # [B'*N*T, C_feat]
        ff = ff.reshape(-1, self.num_clips, self.num_segments, self.in_channels)  # [B', N, T, C_feat]
        ff = ff.transpose(1, 0)  # [N, B', T, C_feat]

        if self.as_neck:
            # naively applying f[clip_orders] produces a tensor shaped [B', N, B', T, C_feat]
            B_ = ff.shape[1]
            shuffle_labels = torch.randint(self.class_num, size=(B_,), device=ff.device)
            clip_orders = torch.tensor(list(permutations(range(self.num_clips))), device=ff.device)[shuffle_labels].T  # [N, B']
            clip_orders = clip_orders.reshape(*clip_orders.shape, 1, 1)  # [N, B', 1, 1]
            clip_orders = clip_orders.expand(-1, -1, *ff.shape[-2:])  # [N, B', T, C_feat], repeated
            ff = ff.gather(dim=0, index=clip_orders)  # [N, B', T, C_feat]

        # P: pair_num = comb(num_clips,2)
        # 2*C_feat is just a reference of original TOL
        pf:list         = [torch.cat([ff[i], ff[j]], dim=-1) for i, j in combinations(range(self.num_clips), 2)]  # dict order, P * [2B, T, 2*C_feat]
        pf:torch.tensor = torch.stack(pf, dim=1)  # [B', P, T, 2*C_feat]
        pf:torch.tensor = pf.reshape(-1, 2*self.in_channels)  # [B'*P*T, 2*C_feat]

        # C_h: num_hidden
        # C: class_num
        h = self.dropout(self.relu(self.fc_cop1(pf)))  # [B'*P*T, C_h]
        h = h.reshape(-1, self.pair_num, self.num_segments, self.num_hidden)  # [B', P, T, C_h]
        h = h.permute(0, 2, 1, 3)  # [B', T, P, C_h]
        h = h.reshape(-1, self.pair_num*self.num_hidden)  # [B'*T, P*C_h]
        h = self.fc_cop2(h)  # logits; [B'*T, C]
        h = h.reshape(-1, self.num_segments, self.class_num)  # [B', T, C]
        h = h.mean(dim=1)  # average consensus; [B', C]

        if self.as_neck:
            loss = {'loss_cop': self.loss_cls(h, shuffle_labels)}
            return f, loss
        else:
            return h

    def loss(self, cls_score, labels, domains, **kwargs):
        losses = super().loss(cls_score, labels)  # todo: dann은 train_ratio 있어도 잘 되는데 왜 여기서만 문제됨?
        return losses
