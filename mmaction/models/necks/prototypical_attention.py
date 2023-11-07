import torch
import torch.nn as nn
import numpy as np
from einops import repeat, reduce, rearrange

from mmcv.cnn import normal_init
from ..builder import NECKS
from ...core import mean_class_accuracy
from mmaction.models.losses import SupConLoss
from mmaction.models import FFNWithNorm


@NECKS.register_module()
class PrototypicalAttention(nn.Module):
    def __init__(self,
                 in_channels,
                 num_clips, num_segments,
                 num_prototypes=32,
                 backbone='TSM',
                 temperature=.07,  # larger the temperature(~=1.), slower the protos learn
                 init_std=0.001,
                 loss_weight=1.,
                 loss_type='self_sim',
                 attention_type='max',  # which embeddings to calculate attention [only max, or all]
    ):
        super().__init__()
        self.in_channels = in_channels
        self.num_clips = num_clips
        self.num_segments = num_segments

        self.num_prototypes = num_prototypes

        self.backbone = backbone
        self.temperature = temperature
        self.init_std = init_std
        self.loss_weight = loss_weight
        self.loss_type = loss_type
        self.attention_type = attention_type

        if self.loss_type == 'infonce':
            self.supcon = SupConLoss(contrast_mode='one')  # view1 = view2
            self.loss_orth = self.infonce  # view1 = view2
        elif self.loss_type == 'self_sim':
            self.loss_orth = self.self_sim

    def init_weights(self):
        self.prototypes = nn.Parameter(torch.randn(self.num_prototypes, self.in_channels))  # [P, C]
        self.prototype_counter = torch.zeros(self.num_prototypes, dtype=torch.int32, requires_grad=False)

    def forward(self, f, labels, domains=None, **kwargs):
        # f: [B x N x T, C, H, W]
        if self.backbone == 'TSM':
            ff = reduce(
                f, '(b n t) c h w -> b n c', 'mean',
                n=self.num_clips, t=self.num_segments).contiguous()
            B, N, C = ff.shape
        assert C == self.in_channels
        losses = dict()

        # N 번째 clip의 P 번째 subaction과의 similarity
        # torch.set_printoptions(linewidth=1000, sci_mode=False)
        # if f.device == torch.device('cuda:0'):
        #     print(self.prototypes[:5,:7])
        prototypes_normed = nn.functional.normalize(self.prototypes, p=2, dim=1)  # [P, C]
        ff_normed = nn.functional.normalize(ff, p=2, dim=2)  # [B, N, C]
        similarity = torch.matmul(ff_normed, prototypes_normed.T)  # [B, N, C], [P, C] -> [B, N, P]

        # compute attention logits
        if self.attention_type == 'max':
            attention_logit, max_args = similarity.max(dim=2)  # [B, N], [B, N]
            self.prototype_counter = self.prototype_counter.to(f.device)
            self.prototype_counter += max_args.reshape(-1).bincount(minlength=self.num_prototypes)
            losses['H_freq'] = self.entropy_from_counter(self.prototype_counter)
        elif self.attention_type == 'all':
            attention_logit = similarity.sum(dim=2)  # [B, N]
        elif self.attention_type == 'subaction':
            attention_logitexp = similarity.div(self.temperature).exp().sum(dim=1)  # [B, P]
            # 합하기 전에 exp 해야 되는 거 아닌가?

        # compute affinity
        if self.attention_type in ['max', 'all']:
            attention = nn.functional.softmax(attention_logit/self.temperature, dim=1)  # [B, N]
            affinity = torch.einsum('bn,bnc->bc', attention, ff)  # [B, C]
        elif self.attention_type in ['subaction']:
            # attention = nn.functional.softmax(attention_logit/self.temperature, dim=1)  # [B, P]
            attention = nn.functional.normalize(attention_logitexp, p=1, dim=1)  # [B, P]
            affinity = torch.einsum('bp,pc->bc', attention, self.prototypes)

        loss_orth = self.loss_orth(prototypes_normed)
        losses['loss_orth'] = self.loss_weight * loss_orth

        return affinity, losses

    def backward_hook(self):
        # Param l2 norm
        pass

    def self_sim(self, prototypes):
        P = prototypes.shape[0]
        self_similarity = torch.matmul(prototypes, prototypes.T)  # [P, P]
        loss_orth = self_similarity.triu(diagonal=1).sum() / (P*(P-1)/2)
        return loss_orth

    def infonce(self, prototypes):  # 같은 feature 복사해서 infonce하면 너무 쉬워서 loss가 0임
        return self.supcon(repeat(prototypes, 'p c -> p 2 c'))  # [P, 2, C], 2 as # views

    @staticmethod
    def entropy_from_counter(counter):
        p = counter / counter.sum()
        weighted_info = -p * torch.log(p+1e-12)
        H = weighted_info.sum()
        return H


@NECKS.register_module()
class ClipAttention(nn.Module):
    def __init__(self,
                 in_channels,
                 num_clips, num_segments,
                 hidden_dim=1024, num_layers=4,
                 backbone='TSM',
                 dropout_ratio=.1,
                 init_std=.001,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.num_clips = num_clips
        self.num_segments = num_segments
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout_ratio = dropout_ratio
        self.init_std = init_std

        self.backbone = backbone

        self.rnn = nn.GRU(
            self.in_channels, self.hidden_dim, self.num_layers,
            batch_first=True, dropout=self.dropout_ratio)
        self.fc_clip_attn = nn.Linear(self.hidden_dim, 1)

    def init_weights(self):
        for name, param in self.rnn.named_parameters():
            if 'weight_ih' in name:
                torch.nn.init.xavier_uniform_(param.data)
            elif 'weight_hh' in name:
                torch.nn.init.orthogonal_(param.data)
            elif 'bias' in name:
                param.data.fill_(0)
        for layer in [self.fc_clip_attn]:
            normal_init(layer, std=self.init_std)

    def forward(self, f, labels, domains=None, **kwargs):
        # f: [B x N x T, C, H, W]
        if self.backbone == 'TSM':
            ff = reduce(
                f, '(b n t) c h w -> b n c', 'mean',
                n=self.num_clips, t=self.num_segments).contiguous()
            B, N, C = ff.shape
        assert C == self.in_channels

        losses = dict()
        fff, _ = self.rnn(ff)  # [B, N, H]
        attention_score = self.fc_clip_attn(rearrange(fff, 'b n h -> (b n) h')).squeeze(dim=1)  # [B x N, H] -> [B x N, 1] -> [B x N]
        attention_score = attention_score.reshape(B, N)  # [B, N]
        attention_score = nn.functional.softmax(attention_score, dim=1)  # [B, N]
        affinity = torch.einsum('bn,bnc->bc', attention_score, ff)  # [B, N], [B, N, C] -> [B, C]
        return affinity, losses
