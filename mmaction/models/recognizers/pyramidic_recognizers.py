from collections.abc import Iterable
from typing import Tuple, List, Union

import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import roc_auc_score
from einops import rearrange, repeat, reduce

from ..builder import RECOGNIZERS
from .base import BaseRecognizer
from .base_da import BaseDARecognizer
from ..common import temporal_locality_fuse


class FrameSampler:
    def __init__(self, sampler_name, sampler_index=dict()):
        self.sampler_name = sampler_name
        self.sampler = getattr(FrameSampler, self.sampler_name)
        assert isinstance(sampler_index, dict)
        self.index_dict = sampler_index

    def __call__(self, imgs):
        return self.sampler(imgs, self.index_dict)

    # def l1(self, imgs):
    #     if index is None:
    #         B, N, C, T, H, W = imgs.shape
    #         index = torch.randint(N, size=(B,), device=imgs.device)
    #         index = repeat(index, 'b -> b 1 c t h w', t=T, c=C, h=H, w=W)
    #         imgs = torch.gather(imgs, dim=1, index=index)
    #     else:
    #         imgs = imgs[:,index:index+1]  # to keep dim
    #     return imgs

    # def g1(self, imgs):
    #     if index is None:
    #         B, N, C, T, H, W = imgs.shape
    #         index = torch.randint(T, size=(B,), device=imgs.device)
    #         index = repeat(index, 'b -> b n c 1 h w', n=N, c=C, h=H, w=W)
    #         imgs = torch.gather(imgs, dim=3, index=index)
    #     else:
    #         imgs = imgs[:,:,:,index:index+1]  # to keep dim
    #     return imgs

    @staticmethod
    def lngn(imgs, index_dict=dict()):
        if index_dict is None:
            B, N, C, T, H, W = imgs.shape
            index_local = torch.randint(N, size=(B,), device=imgs.device)
            index_global = torch.randint(T, size=(B,), device=imgs.device)
            index_local = repeat(index_local, 'b -> b 1 c t h w', t=T, c=C, h=H, w=W)
            index_global = repeat(index_global, 'b -> b n c 1 h w', n=N, c=C, h=H, w=W)
            imgs_local = torch.gather(imgs, dim=1, index=index_local)
            imgs_global = torch.gather(imgs, dim=3, index=index_global)
        else:
            assert isinstance(index_dict, dict)
            index_local:  Union[int,List[int],None] = index_dict.get('l', None)
            index_global: Union[int,List[int],None] = index_dict.get('g', None)
            index_local:  Union[List[int],None]     = [index_local] if isinstance(index_local, int) else index_local
            index_global: Union[List[int],None]     = [index_global] if isinstance(index_global, int) else index_global

            if index_local is not None:
                imgs_locals = []
                for i in index_local:
                    imgs_local = imgs[:,i:i+1]  # to keep dim
                    imgs_locals.append(imgs_local)
                imgs_locals = torch.stack(imgs_locals, dim=1)  # new dim
            else:
                imgs_locals = None

            if index_global is not None:
                imgs_globals = []
                for i in index_global:
                    imgs_global = imgs[:,:,:,i:i+1]  # to keep dim
                    imgs_globals.append(imgs_global)
                imgs_globals = torch.stack(imgs_globals, dim=1)  # new dim
            else:
                imgs_globals = None

        return imgs_locals, imgs_globals

    @staticmethod
    def just_split(imgs, index_dict=dict()):
        """
        imgs: [D x B, V, C, T, H, W]
            note: V indicates the number of clips implying that V clips are already resampled from N clips
        """
        num_locals, num_globals = index_dict.get('l', 0), index_dict.get('g', 0)
        assert num_locals > 0 or num_globals > 0
        imgs_locals, imgs_globals = imgs[:,:num_locals], imgs[:,num_locals:]
        return imgs_locals.unsqueeze(2), imgs_globals.unsqueeze(2)  # to fit to legacy dim `N`


@RECOGNIZERS.register_module()
class TemporallyPyramidicRecognizer(BaseRecognizer):
    def __init__(self,
                 sampler_name:str,
                 sampler_index=dict(),
                 dim='2d',
                 fuse_before_head=True,
                 consensus_before_head=True,
                 locality_aware_head=False,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.sampler_name = sampler_name
        self.sampler = FrameSampler(self.sampler_name, sampler_index)
        self.dim = dim
        self.fuse_before_head = fuse_before_head
        self.consensus_before_head = consensus_before_head
        self.locality_aware_head = locality_aware_head

    def forward_train(self, imgs, labels, **kwargs):
        assert imgs.dim() == 6, 'Use NCTHW'
        self.shape = B, N, C, T, H, W = imgs.shape
        assert N > 1 and T > 1  # N > 1 for uniform sampling, T > 1 for dense sampling
        labels = labels.squeeze(dim=1)  # [B, 1] -> [B], valid for single-labeled

        losses = dict()

        # Extract only of interest features
        imgs_tally:tuple = self.sampler(imgs)
        f_tally = self.extract_feat(imgs_tally)  # [[B, V, NT, C_l], ...]

        if self.with_neck:
            f_tally, losses_aux = self.forward_neck(f_tally, labels, train=True, **kwargs)
            losses.update(losses_aux)

        if self.locality_aware_head:
            cls_score = self.cls_head(f_tally, train=True, **kwargs)
        else:
            f_tally = [f for f in f_tally if f is not None]
            f = torch.cat(f_tally, dim=1)  # [B, sum V, NT, C_l]
            if self.fuse_before_head:
                f = f.mean(dim=1, keepdim=True)
            if self.consensus_before_head:
                f = f.mean(dim=2, keepdim=True)
            f = rearrange(f, 'b vv nt c_l -> (b vv nt) c_l')
            cls_score = self.cls_head(f, train=True, **kwargs)

        if cls_score.shape[0] > B:  # late fuse
            assert cls_score.shape[0] % B == 0
            cls_score = reduce(cls_score, '(b x) k -> b k', 'mean', b=B)

        loss_cls = self.cls_head.loss(cls_score, labels, **kwargs)
        losses.update(loss_cls)
        return losses

    def forward_neck(self, outs, labels, **kwargs):
        losses_aux = dict()
        vertebrae = self.neck
        if not isinstance(vertebrae, Iterable):
            vertebrae = [vertebrae]
        for vertebra in vertebrae:
            outs, loss_aux = vertebra(outs, labels, **kwargs)
            if loss_aux is not None:
                losses_aux.update(loss_aux)
        return outs, losses_aux

    # test methods
    def forward_test(self, imgs):
        """Defines the computation performed at every call when evaluation and
        testing."""
        if self.test_cfg.get('fcn_test', False):
            # If specified, spatially fully-convolutional testing is performed
            assert not self.feature_extraction
            assert self.with_cls_head
            return self._do_fcn_test(imgs).cpu().numpy()
        return self._do_test(imgs).cpu().numpy()

    def _do_test(self, imgs):
        assert imgs.dim() == 6, 'Use NCTHW'
        B, N, C, T, H, W = imgs.shape

        # Extract only of interest features
        imgs_tally:tuple = self.sampler(imgs)
        f_tally = self.extract_feat(imgs_tally)  # [[B, V, NT, C_l], ...]

        if self.with_neck:
            f_tally, _ = self.forward_neck(f_tally, None)

        if self.feature_extraction:
            return reduce(f, 'b vv nt c_l -> b c_l', 'mean')

        if self.locality_aware_head:
            cls_score = self.cls_head(f_tally)
        else:
            f_tally = [f for f in f_tally if f is not None]
            f = torch.cat(f_tally, dim=1)  # [B, sum V, NT, C_l]
            if self.fuse_before_head:
                f = f.mean(dim=1, keepdim=True)
            if self.consensus_before_head:
                f = f.mean(dim=2, keepdim=True)
            f = rearrange(f, 'b vv nt c_l -> (b vv nt) c_l')
            cls_score = self.cls_head(f)

        if cls_score.shape[0] > B:  # late fuse
            assert cls_score.shape[0] % B == 0
            cls_score = reduce(cls_score, '(b x) k -> b k', 'mean', b=B)

        return cls_score

    def forward_dummy(self, imgs, softmax=False):
        pass

    def forward_gradcam(self, imgs):
        """Defines the computation performed at every call when using gradcam
        utils."""
        assert self.with_cls_head
        return self._do_test(imgs)

    def extract_feat(self, imgs_tally):
        f_tally = []
        for imgs, locality in zip(imgs_tally, ['local', 'global']):
            if imgs is None:
                f_tally.append(imgs)
                continue
            _, V, N_, _, T_, _, _ = imgs.shape  # [B, V', N', C, T', H, W], V, N, T would be diff from original after resampling
            if self.dim == '2d':
                imgs = rearrange(imgs, 'b v n c t h w -> (b v n t) c h w').contiguous()  # [B x V x N x T, C, H, W]
                f = super().extract_feat(imgs)  # [B x V x N' x T', C_l, H_l, W_l]
                f = reduce(f, '(b v n t) cl hl wl -> b v (n t) cl', 'mean', v=V, n=N_, t=T_)
            elif self.dim == '3d':
                imgs = rearrange(imgs, 'b v n c t h w -> (b v) c (n t) h w').contiguous()  # [B x V, C, N x T, H, W]
                f = super().extract_feat(imgs)  # [B x V, C_l, NT/8, H_l, W_l]
                f = reduce(f, '(b v) cl nt hl wl -> b v nt cl', 'mean', v=V)
            f_tally.append(f)
        return f_tally


@RECOGNIZERS.register_module()
class TemporallyPyramidicDARecognizer(BaseDARecognizer):
    def __init__(self,
                 sampler_name:str,
                 sampler_index=dict(),
                 dim='2d',
                 fuse_at='mid',
                 *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert fuse_at in ['mid', 'late']
        self.sampler_name = sampler_name
        self.sampler = FrameSampler(self.sampler_name, sampler_index)
        self.fuse_at = fuse_at
        self.dim = dim

    def forward_train(self, imgs_from_domains:List[torch.Tensor], labels, domains, **kwargs):
        """
        Note for nomenclature
        - Fuse: to obtain a single feature vector of a video from multiple clip vectors
        - Consensus: to obtain a single feature vector of a clip from multiple frame vectors
        """
        D = len(imgs_from_domains)
        labels = labels.squeeze(dim=1)  # [D x B, 1] -> [D x B], valid for single-labeled
        losses = dict()

        imgs_from_domains = self.resample(imgs_from_domains)
        f_tallies = self.extract_feat(
            imgs_from_domains,
            consensus=True
        )  # [[source_local, source_global], [target_local, target_global]]

        #################################################################################

        # early fuse (not implemented)

        #################################################################################

        if self.with_neck:
            f_tallies, losses_aux = self.forward_neck(
                f_tallies, labels, domains,
                train=True, **kwargs
            )
            losses.update(losses_aux)

        #################################################################################

        if self.fuse_at == 'mid':
            # mid fuse: fuse + consensus -> head
            outs = temporal_locality_fuse(
                f_tallies,  # [[B, V, C_l], ...]
                temporal_locality='both',
                fusion_method='mean'
            )  # [2 x B. C_l]
            cls_score = self.cls_head(outs, domains=domains, train=True, **kwargs)

        elif self.fuse_at == 'late':
            # late fuse: head -> fuse + consensus
            # [[B, V, N', T', C_l], ...]
            B = f_tallies[0][0].shape[0]
            Vs = [f.shape[1] for f_tally in f_tallies for f in f_tally]
            Vs = np.cumsum(Vs)
            f_tallies_tmp = torch.cat([
                rearrange(f, 'b v n_ t_ c_l -> b (v n_ t_) c_l')
                for f_tally in f_tallies for f in f_tally
            ], dim=1)  # [B x sum V x N' x T', C_l]
            f_tallies_tmp = rearrange(f_tallies_tmp, 'b v_nt c_l -> (b v_nt) c_l')
            cls_score_tallies = self.cls_head(
                f_tallies_tmp, domains=domains, train=True,
                **kwargs
            )  # [B x sum V, K]
            cls_score_tallies = rearrange(
                cls_score_tallies, '(b v_) k -> b v_ k', b=B)  # [B, sum V, K]
            cls_score_tallies = [
                [cls_score_tallies[:,     :Vs[0]], cls_score_tallies[:,Vs[0]:Vs[1]]],
                [cls_score_tallies[:,Vs[1]:Vs[2]], cls_score_tallies[:,Vs[2]:     ]],
            ]  # [[B, V, C_l], ...]
            cls_score = temporal_locality_fuse(
                cls_score_tallies,
                temporal_locality='both',
                fusion_method='mean'
            )

        #################################################################################

        loss_cls = self.cls_head.loss(cls_score, labels, domains=domains, train=True, **kwargs)
        losses.update(loss_cls)
        return losses

    def _do_test(self, imgs, domains=None, **kwargs):
        imgs = self.sampler(imgs)  # [B, N, C, T, H, W] -> [B, V, N', C, T', H, W]
        f_tally = self.extract_feat([imgs], consensus=True)[0]  # assumed single domain

        if self.with_neck:
            f_tally, _ = self.forward_neck(f_tally, None, domains, **kwargs)

        if self.feature_extraction:
            outs = temporal_locality_fuse(
                f_tally,  # [[B, V, C_l], ...]
                temporal_locality='both',
                fusion_method='mean'
            )  # [2 x B, C_l]
            # outs = reduce(outs, '(b n t) c h w -> b c', 'mean', b=B, n=N, t=T)  # [2B x N x T, C, H, W]
            return outs

        f_local, f_global = f_tally  # [B, V1, C_l], [B, V2, C_l]
        V_local = f_local.shape[1]
        subclip_size = self.test_cfg.get('subclip_size', 3)
        outs = []
        for v in list(range(subclip_size, V_local+1, subclip_size)) + ([V_local] if V_local % subclip_size != 0 else []):
            out = torch.cat([f_local[:,v-subclip_size:v], f_global], dim=1).mean(dim=1)  # [B, C_L]
            outs.append(out)
        num_subclips = len(outs)
        outs = rearrange(torch.stack(outs), 'vv b c_l -> (b vv) c_l')  # [B x ceil(V1/3), C_l]

        # should have cls_head if not extracting features
        cls_score = self.cls_head(outs)  # [B x ceil(V1/3), K]
        cls_score = self.average_clip(cls_score, num_segs=num_subclips)  # [B, K]
        return cls_score

    def resample(self, imgs_from_domains):
        resampled = []
        for imgs in imgs_from_domains:
            assert imgs.dim() == 6, 'Use NCTHW'
            B, N, C, T, H, W = imgs.shape
            assert N > 1 and T > 1  # N > 1 for uniform sampling, T > 1 for dense sampling

            # Extract only of-interest features
            imgs:tuple = self.sampler(imgs)  # [B, N, C, T, H, W] -> [B, V, N', C, T', H, W]
            resampled.append(imgs)
        return resampled

    def extract_feat(self, imgs_from_domains, consensus=False):
        f_tallies = []  # [[source_local, source_global], [target_local, target_global]]
        for imgs in imgs_from_domains:
            f_tally = []
            for imgs_subset, locality in zip(imgs, ['local', 'global']):
                if imgs_subset is None:
                    continue
                _, V, N_, _, T_, _, _ = imgs_subset.shape  # [B, V, N', C, T', H_l, W_l], V, N, T would be diff from original after resampling
                # TODO: replace this block in more reasonable logic
                if self.dim == '2d':
                    imgs_subset = rearrange(
                        imgs_subset, 'b v n c t h w -> (b v n t) c h w'
                    ).contiguous()  # [D x B x V x N x T, C, H, W]
                    f = super().extract_feat(imgs_subset)
                    if consensus:
                        f = reduce(f, '(b v n t) cl hl wl -> b v cl', 'mean', v=V, n=N_, t=T_)
                    else:
                        f = reduce(f, '(b v n t) cl hl wl -> b v n t cl', 'mean', v=V, n=N_, t=T_)
                elif self.dim == '3d':
                    imgs_subset = rearrange(
                        imgs_subset, 'b v n c t h w -> (b v) c (n t) h w'
                    ).contiguous()  # [D x B x V x N x T, C, H, W]
                    f = super().extract_feat(imgs_subset)
                    if consensus:
                        f = reduce(f, '(b v) cl nt hl wl -> b v cl', 'mean', v=V)
                    else:
                        f = reduce(f, '(b v) cl nt hl wl -> b v nt cl', 'mean', v=V)

                f_tally.append(f)
            f_tallies.append(f_tally)
        return f_tallies
