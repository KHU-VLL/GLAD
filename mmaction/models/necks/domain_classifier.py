import numpy as np
import torch
import torch.nn as nn
from mmcv.cnn import normal_init

from einops import rearrange, reduce

from ..builder import NECKS
from ..heads.dann_tsm_head import GradReverse
from ..heads.base import get_fc_block_by_channels
from ..common import temporal_locality_fuse


@NECKS.register_module()
class DomainClassifier(nn.Module):
    def __init__(self,
        in_channels,
        loss_weight=1.,
        num_layers=4,
        hidden_dim=4096,
        dropout_ratio=.5,
        init_std=0.001,
        backbone='TSM',  # TSM or TimeSformer
        num_segments=None,  # Only for TSM
        contrastive=False,
        nickname='',  # required if multiple copies of this module are used as neck
        **kwargs
    ):
        super().__init__()
        self.in_channels = in_channels
        self.loss_weight = loss_weight
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.dropout_ratio = dropout_ratio
        self.init_std = init_std
        self.backbone = backbone
        assert self.backbone in ['TSM', 'TimeSformer']
        self.num_segments = num_segments
        self.contrastive = contrastive
        self.nickname = nickname
        self.grl = lambda x: GradReverse.apply(x)
        self.fc_domain:nn.Sequential = get_fc_block_by_channels(
            self.in_channels, 1, [self.hidden_dim]*self.num_layers,
            dropout_ratio=self.dropout_ratio
        )

        self.bce = torch.nn.BCEWithLogitsLoss()

    def init_weights(self):
        """Initiate the parameters from scratch."""
        for layer in self.fc_domain:
            normal_init(layer, std=self.init_std)

    def forward(self, f, labels=None, domains=None, train=False, **kwargs):
        # f:
            # TSM: [2B x N=8 x T=1, n_feat=2048, H=7, W=7]
            # TimeSformer: [2B, n_feat]
        if not train:  # if valid or test
            return f, None

        if self.backbone == 'TSM':
            ff = reduce(
                f, '(bb n t) c h w -> bb c', 'mean',
                n=self.num_segments, t=1)
        else:
            ff = f

        if self.contrastive:
            domains = np.repeat(domains, 2)  # B <- 2B

        source_idx = torch.from_numpy(domains == 'source').to(f.device)  # [2B]
        label_domain = source_idx.type(torch.float32)  # [2B], 1 to source, 0 to target, which to which does not matter
        score_domain = self.fc_domain(self.grl(ff)).squeeze(dim=1)  # [2B]
        loss_domain = self.loss_weight * self.bce(score_domain, label_domain)  # `score_domain` plays a role of logits
        domain_acc = ((score_domain > 0) == label_domain.type(torch.int32)).type(torch.float32).mean()
        suffix = ("_" + self.nickname) if self.nickname else ""
        return f, {
            f'loss_dom{suffix}': loss_domain,
            f'acc_dom{suffix}': domain_acc,
        }  # f: [2B, 1]


@NECKS.register_module()
class TemporallyPyramidicDomainClassifier(DomainClassifier):
    def __init__(self,
        temporal_locality,
        *args, **kwargs
    ):
        self.temporal_locality = temporal_locality
        super().__init__(*args, **kwargs)

    def forward(self,
        f_tallies:list,
        labels=None, domains=None,
        train=False,
        **kwargs
    ):
        if not train:
            return f_tallies, None

        if self.temporal_locality in ['local', 'global', 'both']:
            fs, domains = temporal_locality_fuse(
                f_tallies,
                self.temporal_locality,
                return_domain_names=True
            )

        elif self.temporal_locality == 'local-global':
            fs_source = temporal_locality_fuse(
                [f_tallies[0]], 'local')  # [B, C_l]
            fs_target = temporal_locality_fuse(
                [f_tallies[1]], 'global')  # [B, C_l]
            B = fs_source.shape[0]
            fs = torch.cat([fs_source, fs_target])
            domains = np.array(['source'] * B + ['target'] * B)

        elif self.temporal_locality == 'global-local':
            fs_source = temporal_locality_fuse(
                [f_tallies[0]], 'global')  # [B, C_l]
            fs_target = temporal_locality_fuse(
                [f_tallies[1]], 'local')  # [B, C_l]
            B = fs_source.shape[0]
            fs = torch.cat([fs_source, fs_target])
            domains = np.array(['source'] * B + ['target'] * B)

        elif self.temporal_locality == 'cross':
            fs_source = temporal_locality_fuse(
                [f_tallies[0]], 'local')  # [B, C_l]
            fs_target = temporal_locality_fuse(
                [f_tallies[1]], 'global')  # [B, C_l]
            fs1 = torch.cat([fs_source, fs_target])
            fs_source = temporal_locality_fuse(
                [f_tallies[0]], 'global')  # [B, C_l]
            fs_target = temporal_locality_fuse(
                [f_tallies[1]], 'local')  # [B, C_l]
            fs2 = torch.cat([fs_source, fs_target])
            fs = torch.cat([fs1, fs2])
            B = fs_source.shape[0]
            domains = np.array((['source'] * B + ['target'] * B) * 2)

        _, losses = super().forward(
            fs, domains=domains, train=train, **kwargs)

        return f_tallies, losses
