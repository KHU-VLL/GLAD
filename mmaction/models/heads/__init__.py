# Copyright (c) OpenMMLab. All rights reserved.
from .audio_tsn_head import AudioTSNHead
from .base import BaseHead
from .bbox_head import BBoxHeadAVA
from .fbo_head import FBOHead
from .i3d_head import I3DHead
from .lfb_infer_head import LFBInferHead
from .misc_head import ACRNHead
from .roi_head import AVARoIHead
from .slowfast_head import SlowFastHead
from .ssn_head import SSNHead
from .stgcn_head import STGCNHead
from .timesformer_head import TimeSformerHead
from .tpn_head import TPNHead
from .trn_head import TRNHead
from .tsm_head import TSMHead
from .tsn_head import TSNHead
from .x3d_head import X3DHead
from .osbp_tsm_head import OSBPTSMHead
from .dann_tsm_head import DANNTSMHead
from .contrastiveda_tsm_head import ContrastiveDATSMHead
from .contrastiveda_transformer_head import ContrastiveDATransformerHead
from .tsm_cop_head import TSMTOLHead
from .x3d_cop_head import X3DTOLHead
# from .timesformer_cop_head import TimeSFormerTOLHead
from .timesformer_cop_head import TOLN
from .dino_head import DINODAHead, TSMDINODAHead, DINOHead
from .osbp_tsf_head import OSBPDINODAHead
from .dann_svt_head import DANNDINODAHead
from .places_head import PlacesHead
from .identity import IdentityHead

__all__ = [
    'TSNHead', 'I3DHead', 'BaseHead', 'TSMHead', 'SlowFastHead', 'SSNHead',
    'TPNHead', 'AudioTSNHead', 'X3DHead', 'BBoxHeadAVA', 'AVARoIHead',
    'FBOHead', 'LFBInferHead', 'TRNHead', 'TimeSformerHead', 'ACRNHead',
    'STGCNHead', 'OSBPTSMHead', 'DANNTSMHead', 'ContrastiveDATSMHead', 'TSMTOLHead',
    'X3DTOLHead', 'ContrastiveDATransformerHead', 'DINODAHead',
    'TSMDINODAHead', 'OSBPDINODAHead', 'DANNDINODAHead', 'DINOHead', 'PlacesHead',
    # 'TimeSFormerTOLHead',
    'TOLN',

    'IdentityHead'
]
