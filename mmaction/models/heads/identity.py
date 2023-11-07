from ..builder import HEADS
import torch.nn as nn


@HEADS.register_module()
class IdentityHead(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.identity = nn.Identity()

    def init_weights(self):
        pass

    def forward(self, x, *args, **kwargs):
        x = self.identity(x)
        return x

    def loss(self, cls_score, *args, **kwargs):
        return {}
