import torch
import torch.nn as nn
from mmcv.cnn import normal_init
from torch.autograd import Function

from ..builder import HEADS
from ...core import top_k_accuracy
from .base import AvgConsensus, BaseHead


class GradReverse(Function):
    @staticmethod
    def forward(ctx, x, target_idx_mask):
        ctx.save_for_backward(target_idx_mask)
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        target_idx_mask, = ctx.saved_tensors
        grad_output[target_idx_mask] *= -1.
        return grad_output, None


@HEADS.register_module()
class OSBPTSMHead(BaseHead):
    def __init__(self,
            num_classes,
            in_channels,
            num_layers=1,
            num_segments=8,
            loss_cls=dict(type='OSBPLoss'),
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
        self.num_layers = num_layers
        self.num_segments = num_segments
        self.init_std = init_std
        self.is_shift = is_shift
        self.temporal_pool = temporal_pool

        consensus_ = consensus.copy()

        consensus_type = consensus_.pop('type')
        if consensus_type == 'AvgConsensus':
            self.consensus = AvgConsensus(**consensus_)
        else:
            self.consensus = None

        if self.dropout_ratio != 0:
            self.dropouts = [
                nn.Dropout(p=self.dropout_ratio)
                for _ in range(self.num_layers)
            ]
        else:
            self.dropouts = None
        self.fcs = [
            nn.Linear(self.in_channels//2**i, self.in_channels//2**(i+1))
            for i in range(self.num_layers-1)
        ] + [nn.Linear(self.in_channels//2**(self.num_layers-1), self.num_classes)]
        
        self.fc_block = []
        for i in range(self.num_layers):
            if self.dropout_ratio != 0:
                self.fc_block.append(self.dropouts[i])
            self.fc_block.append(self.fcs[i])
            if i != self.num_layers - 1:
                self.fc_block.append(nn.ReLU())
        self.fc_block = nn.Sequential(*self.fc_block)

        if self.spatial_type == 'avg':
            # use `nn.AdaptiveAvgPool2d` to adaptively match the in_channels.
            self.avg_pool = nn.AdaptiveAvgPool2d(1)
        else:
            self.avg_pool = None

    def init_weights(self):
        """Initiate the parameters from scratch."""
        for layer in self.fc_block:
            normal_init(layer, std=self.init_std)
    
    def forward(self, x, num_segs, domains=None):
        """
        Args:
            x (N x num_segs, c, h, w)
        
        Note:
            N: batch size
            num_segs: num_clips
        """
        if domains is not None and domains.shape[0] > 0:  # if train
            target_idx_mask = torch.squeeze(torch.from_numpy(domains == 'target'))
            target_idx_mask = target_idx_mask.repeat(self.num_segments)
            x = GradReverse.apply(x, target_idx_mask)

        # [N * num_segs, in_channels, 7, 7]
        if self.avg_pool is not None:
            x = self.avg_pool(x)
        # [N * num_segs, in_channels, 1, 1]
        x = torch.flatten(x, 1)
        # [N * num_segs, in_channels]
        cls_score = self.fc_block(x)
        # [N * num_segs, num_classes]
        if self.is_shift and self.temporal_pool:
            # [2 * N, num_segs // 2, num_classes]
            cls_score = cls_score.view((-1, self.num_segments // 2) +
                                       cls_score.size()[1:])
        else:
            # [N, num_segs, num_classes]
            cls_score = cls_score.view((-1, self.num_segments) +
                                       cls_score.size()[1:])
        # [N, 1, num_classes]
        cls_score = self.consensus(cls_score)
        # [N, num_classes]
        return cls_score.squeeze(1)

    def loss(self, cls_score, labels, domains, **kwargs):
        losses = dict()
        if labels.shape == torch.Size([]):
            labels = labels.unsqueeze(0)
        elif labels.dim() == 1 and labels.size()[0] == self.num_classes \
                and cls_score.size()[0] == 1:
            # Fix a bug when training with soft labels and batch size is 1.
            # When using soft labels, `labels` and `cls_socre` share the same
            # shape.
            labels = labels.unsqueeze(0)

        if not self.multi_class and cls_score.size() != labels.size():
            top_k_acc = top_k_accuracy(cls_score.detach().cpu().numpy(),
                                       labels.detach().cpu().numpy(),
                                       self.topk)
            for k, a in zip(self.topk, top_k_acc):
                losses[f'top{k}_acc'] = torch.tensor(
                    a, device=cls_score.device)

        elif self.multi_class and self.label_smooth_eps != 0:
            labels = ((1 - self.label_smooth_eps) * labels +
                      self.label_smooth_eps / self.num_classes)

        loss_cls = self.loss_cls(cls_score, labels, domains, **kwargs)
        # loss_cls may be dictionary or single tensor
        if isinstance(loss_cls, dict):
            losses.update(loss_cls)
        else:
            losses['loss_cls'] = loss_cls

        return losses
