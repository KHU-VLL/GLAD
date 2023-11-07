import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import normal_init

from einops import reduce

from ..builder import NECKS
from ..heads.osbp_tsm_head import GradReverse
from ..heads.base import get_fc_block_by_channels
from ..builder import build_loss
from mmaction.models.common import calc_mca


@NECKS.register_module()
class OSBP(nn.Module):
    def __init__(self,
        in_channels,
        num_classes,  # = K+1
        num_hidden_layers=1,
        dropout_ratio=0.5,
        init_std=0.001,
        target_domain_label=.5,
        loss_weight=1,
        as_head=False,
        backbone='TSM',  # TSM or TimeSformer
        num_segments=None,  # Only for TSM
        **kwargs
    ):
        super().__init__()
        self.in_channels = in_channels
        self.num_hidden_layers = num_hidden_layers
        self.num_classes = num_classes
        self.dropout_ratio = dropout_ratio
        self.init_std = init_std
        self.as_head = as_head
        self.backbone = backbone
        assert self.backbone in ['TSM', 'TimeSformer']
        self.num_segments = num_segments

        self.ffn_osbp:nn.Sequential = nn.Sequential(
            get_fc_block_by_channels(
                self.in_channels, self.in_channels, [self.in_channels] * self.num_hidden_layers,
                bn=False, dropout_ratio=self.dropout_ratio, act_first=True),
            nn.Linear(self.in_channels, self.num_classes)
        )
        self.osbp_loss = build_loss(dict(
            type='OSBPLoss',
            num_classes=num_classes,
            target_domain_label=target_domain_label,
            loss_weight=loss_weight,
        ))

        # for logging
        # a batch may have no target-known
        # to avoid the case
        self.record = {
            'probas': [],
            'labels': [],
        }
        self.record_count = 0
        self.record_length = 20

    def init_weights(self):
        """Initiate the parameters from scratch."""
        for layer in self.ffn_osbp:
            normal_init(layer, std=self.init_std)

    def forward(self, f, labels=None, domains=None, train=False, **kwargs):
        # f
            # TSM: [2B x N=8 x T=1, n_feat, H, W]
            # TimeSformer: [2B x N=1, n_feat=768]
        if self.backbone == 'TSM':
            ff = reduce(f, '(bb n t) c h w -> bb c', 'mean', n=self.num_segments, t=1).contiguous()
        else:
            ff = f

        if self.as_head and not train:  # if this is a head and (valid or test)
            target_idx = torch.ones(ff.shape[0], device=ff.device, dtype=torch.long)
        else:
            if ff.shape[0] == 2*domains.shape[0]:  # if contrastive
                domains = np.repeat(domains, 2)  # N <- 2N
            target_idx = torch.squeeze(torch.from_numpy(domains != 'source'))  # [N]
            ff = GradReverse.apply(ff, target_idx)

        cls_score = self.ffn_osbp(ff)  # [2B, K+1]
        losses = self.osbp_loss(cls_score, labels, domains, **kwargs)

        # logging
        if train:
            source_idx = torch.logical_not(target_idx)
            logits_source = cls_score[source_idx]
            logits_target = cls_score[target_idx]
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
            mca_source = calc_mca(logits_source, labels_source, max_label=self.num_classes)
            mca_target = calc_mca(probas_[:,:-1], labels_, max_label=self.num_classes)
            losses.update({'s_os*': mca_source, 't_os*': mca_target})

            # UNK
            is_unk_gt = labels_ == self.num_classes - 1
            is_unk_pred = probas_.argmax(dim=1) == self.num_classes - 1
            acc_unk = torch.mean((is_unk_gt == is_unk_pred).type(torch.float))
            losses.update({'t_unk': acc_unk})

        if self.as_head:
            return cls_score, losses
        else:
            return f, losses
