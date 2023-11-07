import torch
import torch.nn as nn
import numpy as np
from einops import rearrange, reduce

import math
from itertools import permutations

from mmcv.cnn import normal_init
from ..builder import NECKS
from ...core import mean_class_accuracy


# https://github.com/xudejing/video-clip-order-prediction/blob/master/models/vcopn.py
@NECKS.register_module()
class TOLN(nn.Module):
    """Video temporal order prediction with PFE (Pairwire Feature Extraction), the same as OPN."""
    def __init__(self,
        in_channels,
        num_clips,
        backbone='TimeSformer',  # TSM or TimeSformer
        contrastive=False,
        dropout_ratio=.5,
        init_std=0.001,
        loss_weight=1.,
        num_segments=None,  # Only for TSM
    ):
        """
        Args:
            feature_size (int): 512
        """
        super().__init__()

        self.in_channels = in_channels
        self.num_clips = num_clips
        self.backbone = backbone
        assert self.backbone in ['TSM', 'TimeSformer']
        self.loss_weight = loss_weight
        self.num_segments = num_segments
        self.dropout_ratio = dropout_ratio
        self.init_std = init_std
        self.contrastive = contrastive
        self.num_possible_permutations = math.factorial(num_clips)  # N!

        self.fc_cop1 = nn.Linear(self.in_channels*2, 512)
        pair_num = int(num_clips*(num_clips-1)/2)
        self.fc_cop2 = nn.Linear(512*pair_num, self.num_possible_permutations)

        self.dropout = nn.Dropout(p=self.dropout_ratio)
        self.relu = nn.ReLU(inplace=True)

        self.ce = torch.nn.CrossEntropyLoss()

    def init_weights(self):
        """Initiate the parameters from scratch."""
        for layer in [self.fc_cop1, self.fc_cop2]:
            normal_init(layer, std=self.init_std)

    def forward(self, f, labels=None, domains=None, train=False, **kwargs):
        if not train:  # if valid or test
            return f, None  # f: [B x N=1, n_feat=768]

        # f
            # TSM: [2B x N=3 x T, n_feat, H, W]
            # TimeSformer: [2B x N=3, n_feat=768]
        if self.backbone == 'TSM':
            ff = reduce(
                f, '(bb n t) c h w -> bb n c', 'mean',
                n=self.num_clips, t=self.num_segments).contiguous()
            f_out = reduce(
                f, '(bb n t) c h w -> (bb t) c h w', 'mean',
                n=self.num_clips, t=self.num_segments).contiguous()  # [2B x T, n_feat, H, W]
        else:
            ff = rearrange(f, '(bb n) c -> bb n c', n=self.num_clips).contiguous()  # [2B, N, n_feat]
            f_out = ff.mean(dim=1)  # [2B, n_feat]

        cop_labels = torch.randint(self.num_possible_permutations, size=(ff.shape[0],), device=f.device)  # [2B]
        possible_permutations = torch.tensor(list(permutations(range(self.num_clips))), device=f.device)  # [N!, N]
        arg_permutations = possible_permutations[cop_labels]  # [2B, N], ex: (2, 0, 1)
        arg_permutations = arg_permutations[:,:,None]  # [2B, N, 1], adjust to the same dim as f
        ff = torch.take_along_dim(ff, arg_permutations, dim=1)  # [2B, N, n_feat], same logic as argsort -> take

        ff = rearrange(ff, 'bb n c -> n bb c', n=self.num_clips)  # [N, 2B, n_feat]
        pf = []  # pairwise concat
        for i in range(self.num_clips):
            for j in range(i+1, self.num_clips):
                pf.append(torch.cat([ff[i], ff[j]], dim=1))  # [2B, 2*n_feat]

        pf = [self.fc_cop1(i) for i in pf]
        pf = [self.relu(i) for i in pf]
        h = torch.cat(pf, dim=1)
        h = self.dropout(h)
        h = self.fc_cop2(h)  # logits, [2B, N!]

        loss_cop = self.loss_weight * self.ce(h, cop_labels)
        if domains is not None:
            cop_mca_source, cop_mca_target = self.calc_mca(h, cop_labels, domains)
            losses = {'cop_mca_source': cop_mca_source, 'cop_mca_target': cop_mca_target, 'loss_cop': loss_cop}
        else:
            cop_mca = self.calc_mca(h, cop_labels)
            losses = {'cop_mca': cop_mca, 'loss_cop': loss_cop}
        return f_out, losses

    def calc_mca(self, cls_score, labels, domains=None):
        if domains is not None:  # DA setting
            source_idx = domains == 'source'
            target_idx = ~source_idx
            mca_source = mean_class_accuracy(
                cls_score[source_idx].detach().cpu().numpy(),
                labels[source_idx].detach().cpu().numpy()
            )
            mca_target = mean_class_accuracy(
                cls_score[target_idx].detach().cpu().numpy(),
                labels[target_idx].detach().cpu().numpy()
            )
            scores = (
                torch.tensor(mca_source, device=cls_score.device),
                torch.tensor(mca_target, device=cls_score.device),
            )
            return scores
        else:  # single domain
            score = mean_class_accuracy(
                cls_score.detach().cpu().numpy(),
                labels.detach().cpu().numpy()
            )
            return torch.tensor(score, device=cls_score.device)


@NECKS.register_module()
class TOLN4GLADTOLN):
    def forward(self, f_tallies, labels=None, domains=None, train=False, **kwargs):
        if not train:
            return f_tallies, None

        """f_tallies: [[source_local, source_global], [target_local, target_global]]"""
        # if f_tallies[0] is not None and f_tallies[1] is not None:
        #     f_locals = torch.cat([f_tallies[0][0], f_tallies[1][0]])
        # else:
        #     if f_tallies[0] is not None:
        #         f_locals = f_tallies[0][0]
        #     else:
        if isinstance(f_tallies[0], torch.Tensor) or isinstance(f_tallies[1], torch.Tensor):  # single domain
            f_locals = f_tallies[0]
            if f_locals.dim() == 4:  # [B, V, NT/8, C_l]
                f_locals = f_locals.mean(dim=2)
        else:
            f_locals = torch.cat([f_tallies[0][0], f_tallies[1][0]])
        f_locals = rearrange(f_locals, 'bb n c -> (bb n) c', n=self.num_clips).contiguous()  # [2B x N, n_feat]
        _, loss = super().forward(f_locals, labels=labels, domains=domains, train=train, **kwargs)
        return f_tallies, loss
