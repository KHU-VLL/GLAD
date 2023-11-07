from ..builder import HEADS
from .base import BaseHead

from mmcv.cnn import normal_init

import torch
import torch.nn as nn
from einops import rearrange

import math
from itertools import combinations


# https://github.com/xudejing/video-clip-order-prediction/blob/master/models/vcopn.py
# @HEADS.register_module()
class TOLN(BaseHead):
    """Video temporal order prediction with PFE (Pairwire Feature Extraction), the same as OPN."""
    def __init__(self, base_network, feature_size, tuple_len):
        """
        Args:
            feature_size (int): 512
        """
        super(TOLN, self).__init__()

        self.base_network = base_network
        self.feature_size = feature_size
        self.tuple_len = tuple_len
        self.class_num = math.factorial(tuple_len)

        self.fc_cop1 = nn.Linear(self.feature_size*2, 512)
        pair_num = int(tuple_len*(tuple_len-1)/2)
        self.fc_cop2 = nn.Linear(512*pair_num, self.class_num)

        self.dropout = nn.Dropout(p=0.5)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, tuple):
        f = []  # clip features
        for i in range(self.tuple_len):
            clip = tuple[:, i, :, :, :, :]  # [2B, T, ]
            f.append(self.base_network(clip))

        pf = []  # pairwise concat
        for i in range(self.tuple_len):
            for j in range(i+1, self.tuple_len):
                pf.append(torch.cat([f[i], f[j]], dim=1))

        pf = [self.fc_cop1(i) for i in pf]
        pf = [self.relu(i) for i in pf]
        h = torch.cat(pf, dim=1)
        h = self.dropout(h)
        h = self.fc_cop2(h)  # logits

        return h
