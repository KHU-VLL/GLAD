# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score

from mmcv.cnn import trunc_normal_init

from ..builder import HEADS
from .base import BaseHead


@HEADS.register_module()
class TimeSformerHead(BaseHead):
    """Classification head for TimeSformer.

    Args:
        num_classes (int): Number of classes to be classified.
        in_channels (int): Number of channels in input feature.
        loss_cls (dict): Config for building loss.
            Defaults to `dict(type='CrossEntropyLoss')`.
        init_std (float): Std value for Initiation. Defaults to 0.02.
        kwargs (dict, optional): Any keyword argument to be used to initialize
            the head.
    """

    def __init__(self,
                 num_classes,
                 in_channels,
                 loss_cls=dict(type='CrossEntropyLoss'),
                 init_std=0.02,
                 openset=False,
                 **kwargs):
        super().__init__(num_classes, in_channels, loss_cls, **kwargs)
        self.init_std = init_std
        self.fc_cls = nn.Linear(self.in_channels, self.num_classes)
        self.openset = openset

        # if self.openset:
        #     self.stack_count = 20
        #     self.labels = []
        #     self.preds = []

    def init_weights(self):
        """Initiate the parameters from scratch."""
        trunc_normal_init(self.fc_cls, std=self.init_std)

    def forward(self, x, domains=None, train=False, **kwargs):
        # [N, in_channels]
        cls_score = self.fc_cls(x)
        # [N, num_classes]
        if not train and self.openset:
            threshold = .6
            proba = torch.softmax(cls_score, dim=1)
            H = (-proba * torch.log(proba+1e-12)
                    / np.log(self.num_classes)).sum(dim=1)
            is_unk = H > threshold
            cls_score = torch.cat([cls_score, 100*is_unk.unsqueeze(dim=1)], dim=1)
        return cls_score

    def loss(self, cls_score, labels, domains=None, **kwargs):
        losses = {}
        if domains is not None:
            source_idx = domains == 'source'
            target_idx = ~source_idx
            mca_source = self.calc_mca(cls_score[source_idx], labels[source_idx])
            losses['mca_s'] = mca_source
            if self.openset:
                unk_gt = (labels[target_idx] >= self.num_classes)
                mca_target = self.calc_mca(
                    cls_score[target_idx].detach()[~unk_gt],
                    labels[target_idx][~unk_gt],
                    max_label=self.num_classes)
                losses['mca_os*_t'] = mca_target
                # self.labels.append(labels[target_idx])
                # self.preds.append(cls_score[target_idx].detach())
                # if kwargs['iter'] > self.stack_count:
                #     self.labels = self.labels[1:]
                #     self.preds = self.preds[1:]
                #     labels_ = torch.cat(self.labels)
                #     preds_ = torch.cat(self.preds)
                #     proba = torch.softmax(preds_[:,:self.num_classes], dim=1)
                    # unk_gt = (labels_ >= self.num_classes)
                #     # H_auroc
                #     H = (-proba * torch.log(proba+1e-12)
                #          / np.log(self.num_classes)).sum(dim=1)
                #     H_auc = roc_auc_score(unk_gt.type(torch.int).cpu(), H.cpu())
                    # mca_target = self.calc_mca(preds_[~unk_gt], labels_[~unk_gt], max_label=self.num_classes)
                    # losses['mca_os*_t'] = mca_target
                    # losses['H_auc_t'] = torch.tensor(H_auc, device=cls_score.device)
            else:
                mca_target = self.calc_mca(cls_score[target_idx], labels[target_idx])
                losses['mca_t'] = mca_target
        else:
            mca = self.calc_mca(cls_score, labels)
            losses = {'mca': mca}

        loss_cls = self.loss_cls(cls_score, labels, domains=domains, **kwargs)
        losses['loss_cls'] = loss_cls
        return losses
