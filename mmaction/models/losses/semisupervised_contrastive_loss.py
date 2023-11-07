from ..builder import LOSSES
from .base import BaseWeightedLoss

import torch
from torch.nn.functional import normalize
import numpy as np


@LOSSES.register_module()
class SemisupervisedContrastiveLoss(BaseWeightedLoss):

    def __init__(self, num_classes, unsupervised=True, loss_ratio=.35, tau=.07, loss_weight=1., video_discrimination=False):
        super().__init__(loss_weight)
        self.num_classes = num_classes
        self.unsupervised = unsupervised
        self.loss_ratio = loss_ratio
        self.tau = tau
        self.loss = SupConLoss(self.tau, base_temperature=self.tau)
        self.video_discrimination = video_discrimination

    def _forward(self, cls_score, label, domains=None, **kwargs):
        # cls_score: [2B*2(, N), n_feat]
        # label: [2B*2], [ls1, ls1, ls2, ls2, ..., lt1, lt1, lt2, lt2, ...]
        # domains: [2B]
        cls_score = normalize(cls_score, dim=-1)
        dim = cls_score.dim()
        if self.unsupervised:
            B = cls_score.shape[0] // 4
            if dim == 3:  # [2B*2, N, n_feat]
                N = cls_score.shape[1]
                cls_score = cls_score.reshape(2*B, 2, N, -1)  # [2B, 2, N, n_feat]
                cls_score = cls_score.transpose(1, 2)  # [2B, N, 2, n_feat]
                cls_score = cls_score.reshape(2*B*N*2, -1)  # [2B*N*2, n_feat]
                label = label[:2*B:2].repeat_interleave(N, dim=0)  # [B*N]
                source_view1, source_view2 = cls_score[:2*B*N:2], cls_score[1:2*B*N:2]  # [B*N, n_feat] each
                target_view1, target_view2 = cls_score[2*B*N::2], cls_score[2*B*N+1::2]  # [B*N, n_feat] each
            else:  # [2B*2, n_feat]
                label = label[:2*B:2]  # [B]
                source_view1, source_view2 = cls_score[:2*B:2], cls_score[1:2*B:2]
                target_view1, target_view2 = cls_score[2*B::2], cls_score[2*B+1::2]
            source_input = torch.stack([source_view1, source_view2], dim=1)  # [B*N, 2, n_feat]
            target_input = torch.stack([target_view1, target_view2], dim=1)  # [B*N, 2, n_feat]
            if self.video_discrimination:
                assert dim == 3
                label = label[::N]  # [B]
                source_input = source_input.reshape(B, 2*N, -1)  # [B, N*2, n_feat]
                target_input = target_input.reshape(B, 2*N, -1)  # [B, N*2, n_feat]
            loss_super = self.loss(source_input, label)
            loss_selfsuper = self.loss(source_input)
            loss_selfsuper += self.loss(target_input)
            return {'loss_super': self.loss_ratio * loss_super, 'loss_selfsuper': (1 - self.loss_ratio) * loss_selfsuper}
        else:
            # labels for target domains are pseudo-labels
            loss_super = self.loss(torch.stack([cls_score[::2], cls_score[1::2]], dim=1), label[::2])
            return {'loss_super': loss_super}


class SupConLoss(torch.nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""
    def __init__(self, temperature=0.07, contrast_mode='all',
                 base_temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, labels=None, mask=None):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf
        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """
        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))
        print(log_prob)

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        return loss
