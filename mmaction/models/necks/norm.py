import torch
import torch.nn as nn

from einops import reduce

from ..builder import NECKS


@NECKS.register_module()
class Norm(nn.Module):
    def __init__(self, loss_weight=1., backbone='TSM', num_clips=None, clip_len=None, **kwargs):
        super().__init__()
        self.loss_weight = loss_weight
        assert backbone in ['TSM', 'TimeSformer']
        self.backbone = backbone
        assert backbone != 'TSM' or (num_clips is not None and clip_len is not None)
        self.num_clips = num_clips
        self.clip_len = clip_len

    def init_weights(self):
        pass

    def forward(self, f, labels=None, domains=None, **kwargs):
        # f:
            # TSM: [2B x N=8 x T=1, n_feat=2048, H=7, W=7]
            # TimeSformer: [2B, n_feat]
        if labels is not None:  # if train
            if self.backbone == 'TSM':
                ff = reduce(f, '(bb n t) c h w -> bb c', 'mean', n=self.num_clips, t=self.clip_len)
            loss_norm = torch.norm(ff, dim=1)  # [2B]
            loss_norm = loss_norm.mean()
            return f, {'loss_norm': self.loss_weight * loss_norm}
        else:
            return f, {}
