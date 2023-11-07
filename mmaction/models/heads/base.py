# Copyright (c) OpenMMLab. All rights reserved.
from abc import ABCMeta, abstractmethod

import torch
import torch.nn as nn

from ...core import top_k_accuracy, mean_class_accuracy
from ..builder import build_loss


def get_fc_block(c_in, c_out, num_layers, dropout_ratio):
    fc_block = []
    diff = (c_out - c_in) // num_layers
    c_in_tmp = c_in
    for i in range(num_layers):
        c_out_tmp = c_in + i*diff if i < num_layers - 1 else c_out
        fc = nn.Linear(c_in_tmp, c_out_tmp)
        c_in_tmp = c_out_tmp

        dropout = nn.Dropout(p=dropout_ratio) if dropout_ratio > 0 else nn.Identity()
        act = nn.ReLU() if i < num_layers - 1 else nn.Identity()
        fc_block += [fc, dropout, act]
    return nn.Sequential(*fc_block)


def get_fc_block_by_channels(c_in, c_out, c_mids=[], bn=False, dropout_ratio=.5, act_first=False):
    def get_block(c_in, c_out, bn=False, act=True, dropout_ratio=.5):
        fc = nn.Linear(c_in, c_out)
        bn = nn.BatchNorm1d(c_out) if bn else nn.Identity()
        act = nn.ReLU() if act else nn.Identity()
        dropout = nn.Dropout(p=dropout_ratio) if dropout_ratio > 1e-12 else nn.Identity()
        if act_first:
            return nn.Sequential(fc, bn, act, dropout)
        else:
            return nn.Sequential(fc, bn, dropout, act)
    fc_blocks = nn.ModuleList()
    if c_mids:
        for i, (_c_in, _c_out) in enumerate(zip(
            [c_in] + c_mids,
            c_mids + [c_out]
        )):
            if i == len(c_mids):
                fc_block = get_block(_c_in, _c_out, bn, False, 0)
            else:
                fc_block = get_block(_c_in, _c_out, bn, True, dropout_ratio)
            fc_blocks.append(fc_block)
    else:
        fc_blocks += get_block(c_in, c_out, bn, False, dropout_ratio)
    return nn.Sequential(*fc_blocks)


def concat_unct_to_logits(cls_score, num_classes):
    evidence = torch.exp(torch.clamp(cls_score, -10, 10))  # [D x B, K]
    alpha = evidence + 1  # [D x B, K]
    S = torch.sum(alpha, dim=1, keepdim=True)  # [D x B, 1]
    cls_score = cls_score / S  # [D x B, K]
    uncertainty = num_classes / S  # [D x B, 1]
    cls_score = torch.cat([cls_score, uncertainty], dim=1)  # [D x B, K+1]
    return cls_score


class AvgConsensus(nn.Module):
    """Average consensus module.

    Args:
        dim (int): Decide which dim consensus function to apply.
            Default: 1.
    """

    def __init__(self, dim=1):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        """Defines the computation performed at every call."""
        return x.mean(dim=self.dim, keepdim=True)


class BaseHead(nn.Module, metaclass=ABCMeta):
    """Base class for head.

    All Head should subclass it.
    All subclass should overwrite:
    - Methods:``init_weights``, initializing weights in some modules.
    - Methods:``forward``, supporting to forward both for training and testing.

    Args:
        num_classes (int): Number of classes to be classified.
        in_channels (int): Number of channels in input feature.
        loss_cls (dict): Config for building loss.
            Default: dict(type='CrossEntropyLoss', loss_weight=1.0).
        multi_class (bool): Determines whether it is a multi-class
            recognition task. Default: False.
        label_smooth_eps (float): Epsilon used in label smooth.
            Reference: arxiv.org/abs/1906.02629. Default: 0.
        topk (int | tuple): Top-k accuracy. Default: (1, 5).
    """

    def __init__(self,
                 num_classes,
                 in_channels,
                 loss_cls=dict(type='CrossEntropyLoss', loss_weight=1.0),
                 multi_class=False,
                 label_smooth_eps=0.0,
                 topk=None, print_mca=False):
        super().__init__()
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.loss_cls = build_loss(loss_cls)
        self.multi_class = multi_class
        self.label_smooth_eps = label_smooth_eps
        if topk is not None:
            if isinstance(topk, int):
                topk = (topk,)
        self.topk = topk
        self.print_mca = print_mca

    @abstractmethod
    def init_weights(self):
        """Initiate the parameters either from existing checkpoint or from
        scratch."""

    @abstractmethod
    def forward(self, x):
        """Defines the computation performed at every call."""

    def loss(self, cls_score, labels, domains=None, **kwargs):
        """Calculate the loss given output ``cls_score``, target ``labels``.

        Args:
            cls_score (torch.Tensor): The output of the model.
            labels (torch.Tensor): The target output of the model.

        Returns:
            dict: A dict containing field 'loss_cls'(mandatory)
            and 'topk_acc'(optional).
        """
        losses = dict()
        if labels.shape == torch.Size([]):
            labels = labels.unsqueeze(0)
        elif labels.dim() == 1 and labels.size()[0] == self.num_classes \
                and cls_score.size()[0] == 1:
            # Fix a bug when training with soft labels and batch size is 1.
            # When using soft labels, `labels` and `cls_socre` share the same
            # shape.
            labels = labels.unsqueeze(0)

        if not self.multi_class and cls_score.size() != labels.size() and self.topk is not None:
            top_k_acc = self.calc_topk(cls_score, labels, self.topk, domains=domains)
            for k, a in zip(self.topk, top_k_acc):
                losses[f'top{k}_acc'] = a

        elif self.multi_class and self.label_smooth_eps != 0:
            labels = ((1 - self.label_smooth_eps) * labels +
                      self.label_smooth_eps / self.num_classes)

        if self.print_mca:
            mca = self.calc_mca(cls_score, labels)
            losses['mca'] = mca

        loss_cls = self.loss_cls(cls_score, labels, domains=domains, **kwargs)
        # loss_cls may be dictionary or single tensor
        if isinstance(loss_cls, dict):
            losses.update(loss_cls)
        else:
            losses['loss_cls'] = loss_cls

        return losses

    def calc_topk(self, cls_score, labels, topk, domains=None):
        if domains is not None:
            source_idx = domains == 'source'
            cls_score = cls_score[source_idx]
            labels = labels[source_idx]
        cls_score = cls_score[0] if type(cls_score) == list else cls_score  # list: dann
        top_k_acc = top_k_accuracy(
            cls_score.detach().cpu().numpy(),
            labels.detach().cpu().numpy(),
            topk)
        return [torch.tensor(acc, device=cls_score.device) for acc in top_k_acc]

    def calc_mca(self, cls_score, labels, max_label=-1):
        mca = mean_class_accuracy(
            cls_score.detach().cpu().numpy(),
            labels.detach().cpu().numpy(),
            max_label=max_label
        )
        return torch.tensor(mca, device=cls_score.device)

    def unflatten_based_on_shiftedness(self, x):
        if self.is_shift and self.temporal_pool:
            # [2 * N, num_segs // 2, num_classes]
            return x.view((-1, self.num_segments // 2) + x.size()[1:])
        else:
            # [N, num_segs, num_classes]
            return x.view((-1, self.num_segments) + x.size()[1:])


# 이게 필요한가? 어차피 domains를 kwargs로 처리하는데
class BaseDAHead(BaseHead):
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

        if self.multi_class and self.label_smooth_eps != 0:
            labels = ((1 - self.label_smooth_eps) * labels +
                      self.label_smooth_eps / self.num_classes)

        loss_cls = self.loss_cls(cls_score, labels, domains, **kwargs)

        # evaluating train scores
        # if not self.multi_class and cls_score.size() != labels.size() and self.topk is not None:
        #     top_k_acc = self.calc_topk(cls_score, labels, self.topk)
        #     for k, a in zip(self.topk, top_k_acc):
        #         losses[f'top{k}_acc'] = a

        if self.print_mca:
            mca = self.calc_mca(cls_score, labels)
            losses['mca'] = mca

        # loss_cls may be dictionary or single tensor
        if isinstance(loss_cls, dict):
            losses.update(loss_cls)
        else:
            losses['loss_cls'] = loss_cls

        return losses


class BaseDAContrastiveHead(BaseHead):
    def loss(self, cls_score, labels, domains, **kwargs):
        """Calculate the loss given output ``cls_score``, target ``labels``.

        Args:
            cls_score (torch.Tensor): The output of the model. [2B*2(, N), 1~3, n_feat]
            labels (torch.Tensor): The target output of the model.

        Returns:
            dict: A dict containing field 'loss_cls'(mandatory)
            and 'topk_acc'(optional).
        """

        if labels.shape == torch.Size([]):
            labels = labels.unsqueeze(0)
        elif labels.dim() == 1 and labels.size()[0] == self.num_classes and cls_score.size()[0] == 1:
            # Fix a bug when training with soft labels and batch size is 1.
            # When using soft labels, `labels` and `cls_socre` share the same
            # shape.
            labels = labels.unsqueeze(0)

        # if self.linear_head_debug:
        #     return {'loss_linear_head_debug': self.ce(cls_score[:,0,:], labels)}

        losses = dict()
        B = cls_score.shape[0] // 4
        num_labeled = 2*B if self.loss_cls.unsupervised else -1

        if self.print_mca:
            # centroids are needed only for scoring
            self._update_centroids(cls_score[:num_labeled], labels[:num_labeled])
            # log train scores
            if not self.multi_class and cls_score.size() != labels.size():  # using hard label
                losses['mca'] = self._get_train_mca(cls_score[:num_labeled], labels[:num_labeled])

        if self.multi_class and self.label_smooth_eps != 0:
            labels = self._smoothen_labels(labels)

        if cls_score.dim() == 4:  # [2B*2, N, 1, n_feat]
            loss_cls = self.loss_cls(cls_score[:,:,0,:], labels, domains, **kwargs)
        elif cls_score.dim() == 3:  # [2B*2, 1~3, n_feat]
            loss_cls = [self.loss_cls(cls_score[:,i], labels, domains, **kwargs) for i in range(cls_score.shape[1])]  # list of dicts
            loss_cls = {k: sum([l[k] for l in loss_cls]) for k in ['loss_super', 'loss_selfsuper']}  # sum each loss
        elif cls_score.dim() == 2:  # [2B*2, n_feat]
            loss_cls = self.loss_cls(cls_score, labels, domains, **kwargs)
        losses.update(loss_cls)

        # kill as soon as any of losses becomes NaN
        if any(losses[loss_name].isnan() for loss_name in [loss_name for loss_name in losses.keys() if 'loss_' in loss_name]):
            # kill the parent process
            import os
            import signal
            print('\n\n\nkill this training process due to nan values\n\n\n')
            os.kill(os.getppid(), signal.SIGTERM)

        return losses

    def _update_centroids(self, features, labels):
        if not self.with_given_centroids:
            with torch.no_grad():
                for c in range(self.num_classes):
                    c_features = features[labels==c]  # [B*2(, N)(, 1~3), n_feat]
                    if features.dim() == 4:
                        c_features = c_features[:,:,0].reshape(-1, features.shape[-1])  # [B*2*N, n_feat]
                    elif features.dim() == 3:
                        c_features = c_features[:,0]
                    self.centroids[c].update(c_features)

    def _get_train_mca(self, cls_score, labels):
        with torch.no_grad():
            logits = self._get_logits_from_features(cls_score)  # [B*2, C]
            mca = mean_class_accuracy(
                logits.detach().cpu().numpy(),  # score := negative distance
                labels.detach().cpu().numpy()
            )
            return torch.tensor(mca, device=logits.device)

    def _get_logits_from_features(self, features):
        with torch.no_grad():
            if features.dim() == 4:  # [B*2, N, 1~3, n_feat]
                features = features[:,0]  # [B*2, 1~3, n_feat]
            elif features.dim() == 2:  # [B, n_feat]  (test, or transformers)
                features = features.unsqueeze(dim=1)  # [B, 1, n_feat]
            features = features[:,:1]  # [B*2, 1, n_feat]
            centroids = (
                self.centroids if self.with_given_centroids
                else torch.stack([c.mean for c in self.centroids])
            )  # [C, n_feat]
            centroids = centroids.unsqueeze(dim=0)   # [1,   C, n_feat]
            distances = (features - centroids) ** 2  # [B*2, C, n_feat]
            distances = distances.mean(dim=2) ** .5  # [B*2, C]
            return -distances

    def _smoothen_labels(self, labels):
        smooth_labels = (
            (1 - self.label_smooth_eps) * labels
            + self.label_smooth_eps / self.num_classes
        )
        return smooth_labels

    def _get_loss(self, cls_score, labels, domains, **kwargs):
        pass
