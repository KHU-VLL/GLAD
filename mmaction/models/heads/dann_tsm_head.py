import torch
import torch.nn as nn
from mmcv.cnn import normal_init
from torch.autograd import Function
import numpy as np

from ..builder import HEADS, build_loss
from ...core import top_k_accuracy
from .base import AvgConsensus, BaseHead

from itertools import chain


class GradReverse(Function):
    @staticmethod
    def forward(ctx, x):
        return x

    @staticmethod
    def backward(ctx, grad_output):
        grad_output *= -1.
        return grad_output


@HEADS.register_module()
class DANNTSMHead(BaseHead):
    def __init__(self,
            num_classes,
            in_channels,
            num_cls_layers=1,
            num_domain_layers=4,
            num_segments=8,
            loss_cls=dict(type='CrossEntropyLoss'),
            loss_domain=dict(type='DANNDomainLoss', loss_weight=.5),
            spatial_type='avg',
            consensus=dict(type='AvgConsensus', dim=1),
            dropout_ratio=0.8,
            init_std=0.001,
            is_shift=True,
            temporal_pool=False,
            **kwargs):
        super().__init__(num_classes, in_channels, loss_cls, **kwargs)

        self.spatial_type = spatial_type
        self.dropout_ratio = dropout_ratio
        self.num_cls_layers = num_cls_layers
        self.num_domain_layers = num_domain_layers
        self.num_segments = num_segments
        self.init_std = init_std
        self.is_shift = is_shift
        self.temporal_pool = temporal_pool

        self.loss_domain = build_loss(loss_domain)

        consensus_ = consensus.copy()
        consensus_type = consensus_.pop('type')

        if consensus_type == 'AvgConsensus':
            self.consensus_cls = AvgConsensus(**consensus_)
            self.consensus_domain = AvgConsensus(**consensus_)
        else:
            self.consensus_cls = nn.Identity()
            self.consensus_domain = nn.Identity()
        
        def get_fc_block(c_in, c_out, num_layers):
            fc_block = []
            for i in range(num_layers):
                if i < num_layers - 1:
                    c_out_ = c_in // 2**(i+1)
                    fc = nn.Linear(c_in//2**i, c_out_)
                else:
                    c_out_= c_out
                    fc = nn.Linear(c_in//2**(num_layers-1), c_out_)
                dropout = nn.Dropout(p=self.dropout_ratio) if self.dropout_ratio > 0 else nn.Identity()
                act = nn.ReLU() if i < num_layers - 1 else nn.Identity()
                fc_block += [fc, dropout, act]
            return nn.Sequential(*fc_block)
        
        self.avg_pool = nn.AdaptiveAvgPool2d(1) if self.spatial_type == 'avg' else nn.Identity()
        self.grl = lambda x: GradReverse.apply(x)
        self.fc_cls = get_fc_block(self.in_channels, self.num_classes, self.num_cls_layers)
        self.fc_domain = get_fc_block(self.in_channels, 1, self.num_domain_layers)

    def init_weights(self):
        """Initiate the parameters from scratch."""
        for layer in chain(self.fc_cls, self.fc_domain):
            normal_init(layer, std=self.init_std)
    
    def forward(self, f, num_segs, domains=None):
        """
        Args:
            f (N x num_segs, c, h, w)
        
        Note:
            N: batch size
            num_segs: num_clips
        """

        # [N*segs, C_in, 7, 7]
        f = self.avg_pool(f)  # [N*segs, C_in, 1, 1]
        f = torch.flatten(f, start_dim=1)  # [N*segs, C_in]
        if domains is not None and domains.shape[0] > 0:  # if train
            self.idx_classified = (domains == 'source')
        else:  # if val
            self.idx_classified = np.ones(f.shape[0]//self.num_segments, dtype=bool)  # all inputs have labels => can be classified
        repeated_idx_classified = np.repeat(self.idx_classified, self.num_segments)  # same as torch.repeat_interleave
        cls_score = self.fc_cls(f[repeated_idx_classified])  # [N*segs, K]
        domain_score = self.fc_domain(self.grl(f))  # [N*segs, 1]
    
        cls_score = self.unflatten_based_on_shiftedness(cls_score)  # [N, segs, K] or [2*N, segs//2, K] (not shifted)
        domain_score = self.unflatten_based_on_shiftedness(domain_score)  # [N, segs, 1] or [2*N, segs//2, 1] (not shifted)
        
        cls_score = self.consensus_cls(cls_score)  # [N, 1, K]
        domain_score = self.consensus_domain(domain_score)  # [N, 1, 1]
        return [cls_score.squeeze(dim=1), domain_score.squeeze(dim=1)]  # [N, num_classes], [N, 1]

    def loss(self, cls_score, labels, domains, **kwargs):
        losses = dict()
        labels = labels[self.idx_classified]
        if labels.shape == torch.Size([]):
            labels = labels.unsqueeze(0)
        elif labels.dim() == 1 and labels.size()[0] == self.num_classes \
                and cls_score[0].size()[0] == 1:
            # Fix a bug when training with soft labels and batch size is 1.
            # When using soft labels, `labels` and `cls_socre` share the same
            # shape.
            labels = labels.unsqueeze(0)

        if not self.multi_class and cls_score[0].size() != labels.size():
            top_k_acc = top_k_accuracy(cls_score[0].detach().cpu().numpy(),
                                       labels.detach().cpu().numpy(),
                                       self.topk)
            for k, a in zip(self.topk, top_k_acc):
                losses[f'top{k}_acc'] = torch.tensor(
                    a, device=cls_score[0].device)

        elif self.multi_class and self.label_smooth_eps != 0:
            labels = ((1 - self.label_smooth_eps) * labels +
                      self.label_smooth_eps / self.num_classes)

        cls_score, domain_score = cls_score
        loss_cls = self.loss_cls(cls_score, labels, domains, **kwargs)
        loss_domain = self.loss_domain(domain_score, labels, domains, **kwargs)
        # loss_cls may be dictionary or single tensor
        for loss in [loss_cls, loss_domain]:
            if isinstance(loss, dict):
                losses.update(loss)  # loss_domain
            else:
                losses['loss_cls'] = loss
        
        return losses

    def unflatten_based_on_shiftedness(self, x):
        if self.is_shift and self.temporal_pool:
            # [2 * N, num_segs // 2, num_classes]
            return x.view((-1, self.num_segments // 2) + x.size()[1:])
        else:
            # [N * num_segs, *] -> [N, num_segs, *]
            return x.view((-1, self.num_segments) + x.size()[1:])
