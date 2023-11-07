# Copyright (c) OpenMMLab. All rights reserved.
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import normal_init

from einops import rearrange, reduce

from ..builder import HEADS
from .base import AvgConsensus, BaseHead, concat_unct_to_logits
from mmaction.models.common import calc_mca


@HEADS.register_module()
class TSMHead(BaseHead):
    """Class head for TSM.

    Args:
        num_classes (int): Number of classes to be classified.
        in_channels (int): Number of channels in input feature.
        num_segments (int): Number of frame segments. Default: 8.
        loss_cls (dict): Config for building loss.
            Default: dict(type='CrossEntropyLoss')
        spatial_type (str): Pooling type in spatial dimension. Default: 'avg'.
        consensus (dict): Consensus config dict.
        dropout_ratio (float): Probability of dropout layer. Default: 0.4.
        init_std (float): Std value for Initiation. Default: 0.01.
        is_shift (bool): Indicating whether the feature is shifted.
            Default: True.
        temporal_pool (bool): Indicating whether feature is temporal pooled.
            Default: False.
        kwargs (dict, optional): Any keyword argument to be used to initialize
            the head.
    """

    def __init__(self,
                 num_classes,
                 in_channels,
                 num_segments=8,
                 loss_cls=dict(type='CrossEntropyLoss'),
                 spatial_type='avg',
                 consensus=dict(type='AvgConsensus', dim=1),
                 dropout_ratio=0.5,
                 init_std=0.001,
                 is_shift=True,
                 temporal_pool=False,
                 openset=False,
                 **kwargs):
        super().__init__(num_classes, in_channels, loss_cls, **kwargs)

        self.spatial_type = spatial_type
        self.dropout_ratio = dropout_ratio
        self.num_segments = num_segments
        self.init_std = init_std
        self.is_shift = is_shift
        self.temporal_pool = temporal_pool
        self.openset = openset

        if self.openset:
            self.record = {
                'probas': [],
                'labels': [],
            }
            self.record_count = 0
            self.record_length = 20

        consensus_ = consensus.copy()

        consensus_type = consensus_.pop('type')
        if consensus_type == 'AvgConsensus':
            self.consensus = AvgConsensus(**consensus_)
        else:
            self.consensus = None

        if self.dropout_ratio != 0:
            self.dropout = nn.Dropout(p=self.dropout_ratio)
        else:
            self.dropout = None
        self.fc_cls = nn.Linear(self.in_channels, self.num_classes)

        if self.spatial_type == 'avg':
            # use `nn.AdaptiveAvgPool2d` to adaptively match the in_channels.
            self.avg_pool = nn.AdaptiveAvgPool2d(1)
        else:
            self.avg_pool = None

    def init_weights(self):
        """Initiate the parameters from scratch."""
        normal_init(self.fc_cls, std=self.init_std)

    def forward(self, x, num_segs=None, domains=None, train=False, **kwargs):
        """Defines the computation performed at every call.

        Args:
            x (torch.Tensor): The input data.
            num_segs (int): Useless in TSMHead. By default, `num_segs`
                is equal to `clip_len * num_clips * num_crops`, which is
                automatically generated in Recognizer forward phase and
                useless in TSM models. The `self.num_segments` we need is a
                hyper parameter to build TSM models.
        Returns:
            torch.Tensor: The classification scores for input samples.
        """
        # N or T is 1 during training time
        is_pooled = x.dim() != 4
        if not is_pooled:
            # [2B x N x T, in_channels, 7, 7]
            if self.avg_pool is not None:
                x = self.avg_pool(x)
            # [2B x N x T, in_channels, 1, 1]
            x = torch.flatten(x, 1)

        # [2B x N x T, in_channels]
        if self.dropout is not None:
            x = self.dropout(x)
        # [2B x N x T, num_classes]
        cls_score = self.fc_cls(x)

        if not is_pooled:
            if self.is_shift and self.temporal_pool:
                # [2B x 2, T // 2, num_classes]
                cls_score = cls_score.view((-1, self.num_segments // 2) +
                                        cls_score.size()[1:])
            else:
                # [2B, N x T, num_classes]  (N or T is 1)
                cls_score = cls_score.view((-1, self.num_segments) +
                                        cls_score.size()[1:])
            # [2B, 1, num_classes]
            cls_score = self.consensus(cls_score)
        cls_score = cls_score.squeeze(1)  # [D x B, K]

        if not train and self.openset:
            cls_score = concat_unct_to_logits(cls_score, self.num_classes)
            # evidence = torch.exp(torch.clamp(cls_score, -10, 10))  # [D x B, K]
            # alpha = evidence + 1  # [D x B, K]
            # S = torch.sum(alpha, dim=1, keepdim=True)  # [D x B, 1]
            # cls_score = cls_score / S  # [D x B, K]
            # uncertainty = self.num_classes / S  # [D x B, 1]
            # cls_score = torch.cat([cls_score, uncertainty], dim=1)  # [D x B, K+1]

        return cls_score

    def loss(self, cls_score, labels, domains=None, train=False, **kwargs):
        losses = {}
        if domains is not None:
            source_idx = domains == 'source'
            target_idx = ~source_idx
            mca_source = self.calc_mca(cls_score[source_idx].detach(), labels[source_idx])
            if self.openset:
                logits = concat_unct_to_logits(cls_score.detach(), self.num_classes)
                logits_source = logits[source_idx]
                logits_target = logits[target_idx]
                labels_source = labels[source_idx]
                labels_target = labels[target_idx]
                probas_target = F.softmax(logits_target, dim=1)  # [B, K+1]
                self.record['probas'].append(probas_target)
                self.record['labels'].append(labels_target)
                if self.record_count > self.record_length:
                    self.record['probas'] = self.record['probas'][1:]
                    self.record['labels'] = self.record['labels'][1:]
                self.record_count += 1
                probas_ = torch.cat(self.record['probas'])
                labels_ = torch.cat(self.record['labels'])

                # OS*
                mca_source = calc_mca(logits_source, labels_source, max_label=self.num_classes+1)
                mca_target = calc_mca(probas_, labels_, max_label=self.num_classes+1)
                losses.update({'s_os*': mca_source, 't_os*': mca_target})

                # UNK
                is_unk_gt = labels_ == self.num_classes
                is_unk_pred = probas_.argmax(dim=1) == self.num_classes
                acc_unk = torch.mean((is_unk_gt == is_unk_pred).type(torch.float))
                losses.update({'t_unk': acc_unk})
            else:
                mca_target = self.calc_mca(cls_score[target_idx], labels[target_idx])
                losses['t_mca'] = mca_source
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


@HEADS.register_module()
class TemporalLocalityAwareTSMHead(BaseHead):
    def __init__(self,
                 num_classes,
                 in_channels,
                 hidden_dim=2048,
                 mid_dim=2048,
                 multi_domain=False,
                 num_segments=8,
                 loss_cls=dict(type='CrossEntropyLoss'),
                 dropout_ratio=0.5,
                 init_std=0.001,
                 **kwargs):
        super().__init__(num_classes, in_channels, loss_cls, **kwargs)

        self.hidden_dim = hidden_dim
        self.mid_dim = mid_dim
        self.multi_domain = multi_domain
        self.dropout_ratio = dropout_ratio
        self.num_segments = num_segments
        self.init_std = init_std

        self.norm2_local = nn.LayerNorm(self.in_channels)
        self.norm2_global = nn.LayerNorm(self.in_channels)
        self.mlp_local = self.build_block(
            self.in_channels, self.hidden_dim, self.mid_dim,
            self.dropout_ratio
        )
        self.mlp_global = self.build_block(
            self.in_channels, self.hidden_dim, self.mid_dim,
            self.dropout_ratio
        )
        self.fc_tl_aware_cls = nn.Linear(2*self.mid_dim, self.num_classes)

    def init_weights(self):
        """Initiate the parameters from scratch."""
        for layer in self.mlp_local:
            if isinstance(layer, nn.Linear):
                normal_init(layer, std=self.init_std)
        for layer in self.mlp_global:
            if isinstance(layer, nn.Linear):
                normal_init(layer, std=self.init_std)
        normal_init(self.fc_tl_aware_cls, std=self.init_std)

    @staticmethod
    def build_block(
        input_dim, hidden_dim, out_dim,
        dropout_ratio,
    ):
        return nn.Sequential(OrderedDict([
            ('drop0', nn.Dropout(p=dropout_ratio)),
            ('ffn0', nn.Linear(input_dim, hidden_dim)),
            ('act', nn.GELU()),
            ('ffn1', nn.Linear(hidden_dim, out_dim)),
            ('drop1', nn.Dropout(p=dropout_ratio)),
        ]))

    def forward(self, *args, **kwargs):
        if self.multi_domain:
            return self.forward_multi_domain(*args, **kwargs)
        else:
            return self.forward_single_domain(*args, **kwargs)

    def forward_single_domain(self, f_tally, train=False, **kwargs):
        """
        # params
        - `f_tally`:
            - shape: `[[B, V0, NT, C], [B, V1, NT, C]]`
            - meaning: `[local, global]`
            - one of element can be `None`
        """
        fs = []
        B = 0
        device = None
        for f, mlp, norm in zip(
            f_tally,
            [self.mlp_local, self.mlp_global],
            [self.norm2_local, self.norm2_global]
        ):
            if f is None:
                fs.append(None)
                continue
            B, device = f.shape[0], f.device
            f = rearrange(f, 'b v nt c -> (b v nt) c')
            f = f + mlp(norm(f))
            fs.append(f)
        fs = [
            reduce(f, '(b x) c_mid -> b c_mid', 'mean', b=B)
                if f is not None
                else torch.zeros((B, self.mid_dim), device=device)
            for f in fs
        ]
        f = torch.cat(fs, dim=1)  # [B, 2 x C_mid]
        cls_score = self.fc_tl_aware_cls(f)
        return cls_score

    def forward_multi_domain(self, f_tallies, domains, train=False, **kwargs):
        pass

    def loss(self, cls_score, labels, domains=None, **kwargs):
        if domains is not None:
            source_idx = domains == 'source'
            target_idx = ~source_idx
            mca_source = self.calc_mca(cls_score[source_idx], labels[source_idx])
            mca_target = self.calc_mca(cls_score[target_idx], labels[target_idx])
            losses = {'mca_source': mca_source, 'mca_target': mca_target}
        else:
            mca = self.calc_mca(cls_score, labels)
            losses = {'mca': mca}

        loss_cls = self.loss_cls(cls_score, labels, domains=domains, **kwargs)
        losses['loss_cls'] = loss_cls
        return losses
