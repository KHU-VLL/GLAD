from collections.abc import Iterable

import torch
from torch import nn
import numpy as np
from einops import reduce

from ..builder import RECOGNIZERS
from .recognizer2d import DARecognizer2D


@RECOGNIZERS.register_module()
class DARecognizer3D(DARecognizer2D):
    def forward_train(self, imgs, labels, domains:np.ndarray, **kwargs):
        """Defines the computation performed at every call when training."""

        assert self.with_cls_head
        if self.contrastive:
            imgs = imgs.reshape((-1, ) + imgs.shape[-4:])  # [2B, 2, 1, C, T, H, W] -> [4B, C, T, H, W]
            labels = labels.reshape(-1)  # [2B, 2, 1] -> [4B]
            domains = np.tile(domains, (2, 1)).T.reshape(-1)  # [2B] -> [2, 2B] -> [2B, 2] -> [4B]
        else:
            # TOL: [2B, N, C, T, H, W]
            if isinstance(imgs, list):
                new_imgs = []
                for imgs_ in imgs:
                    if imgs_.dim() > 5:
                        imgs_ = imgs_.reshape((-1, ) + imgs_.shape[2:])  # -> [B x N, C, T, H, W]
                        new_imgs.append(imgs_)
                imgs = new_imgs
            elif imgs.dim() > 5:
                imgs = imgs.reshape((-1, ) + imgs.shape[2:])  # -> [2B x N, C, T, H, W]
            else:
                pass
        losses = dict()

        if isinstance(imgs, list):
            x_source, x_target = self.extract_feat(imgs)
            B = (domains == 'source').sum()
            x_source = reduce(x_source, '(b n) cl tl hl wl -> b cl', 'mean', b=B).contiguous()
            x_target = reduce(x_target, '(b n) cl tl hl wl -> b cl', 'mean', b=B).contiguous()
            x = torch.cat([x_source, x_target])
            # if self.with_neck:
            #     x, losses_aux = self.forward_neck(x, labels.squeeze(dim=1), train=True, domains=domains, **kwargs)
            #     losses.update(losses_aux)
        else:
            x = self.extract_feat(imgs)
            # BB = 2*(domains == 'source').sum()
            # x = reduce(x, '(bb n) cl tl hl wl -> bb cl', 'mean', bb=BB).contiguous()  # TOL does not work with this code
            x = reduce(x, 'bbn cl tl hl wl -> bbn cl', 'mean').contiguous()

        if self.with_neck:
            x, losses_aux = self.forward_neck(x, labels.squeeze(dim=1), train=True, domains=domains, **kwargs)
            losses.update(losses_aux)

        gt_labels = labels.squeeze(dim=1)
        cls_score = self.cls_head(x, labels=gt_labels, domains=domains, train=True, **kwargs)
        loss_cls = self.cls_head.loss(cls_score, labels=gt_labels, domains=domains, train=True, **kwargs)
        losses.update(loss_cls)

        return losses

    def extract_feat(self, imgs):
        if isinstance(imgs, list):
            imgs_source, imgs_target = imgs
            x_source = super().extract_feat(imgs_source)
            x_target = super().extract_feat(imgs_target)
            return [x_source, x_target]
        else:
           return super().extract_feat(imgs)

    def forward_neck(self, x, labels, **kwargs):
        losses_aux = dict()
        vertebrae = self.neck
        if not isinstance(vertebrae, Iterable):
            vertebrae = [vertebrae]
        for vertebra in vertebrae:
            if isinstance(x, list):
                x_source, x_target = x
                x_source, loss_aux_source = vertebra(x_source, labels, **kwargs)
                x_target, loss_aux_target = vertebra(x_target, labels, **kwargs)
                loss_aux_source = {k+'_s': v for k, v in loss_aux_source.items()}
                loss_aux_target = {k+'_t': v for k, v in loss_aux_target.items()}
                losses_aux.update(loss_aux_source)
                losses_aux.update(loss_aux_target)
            else:
                x, loss_aux = vertebra(x, labels, **kwargs)
                losses_aux.update(loss_aux)
        return x, losses_aux

    def _do_test(self, imgs, domains):
        """Defines the computation performed at every call when evaluation,
        testing and gradcam."""
        batches = imgs.shape[0]
        num_segs = imgs.shape[1]
        imgs = imgs.reshape((-1, ) + imgs.shape[2:])

        if self.max_testing_views is not None:
            total_views = imgs.shape[0]
            assert num_segs == total_views, (
                'max_testing_views is only compatible '
                'with batch_size == 1')
            view_ptr = 0
            feats = []
            while view_ptr < total_views:
                batch_imgs = imgs[view_ptr:view_ptr + self.max_testing_views]
                x = self.extract_feat(batch_imgs)
                if self.with_neck:
                    x, _ = self.neck(x)
                feats.append(x)
                view_ptr += self.max_testing_views
            # should consider the case that feat is a tuple
            if isinstance(feats[0], tuple):
                len_tuple = len(feats[0])
                feat = [
                    torch.cat([x[i] for x in feats]) for i in range(len_tuple)
                ]
                feat = tuple(feat)
            else:
                feat = torch.cat(feats)
        else:
            feat = self.extract_feat(imgs)
            feat = reduce(feat, 'bn cl tl hl wl -> bn cl', 'mean').contiguous()
            if self.with_neck:
                vertebrae = self.neck
                if not isinstance(vertebrae, Iterable):
                    vertebrae = [vertebrae]
                for vertebra in vertebrae:
                    feat, _ = vertebra(feat, None, domains=domains)

        if self.feature_extraction:
            feat_dim = len(feat[0].size()) if isinstance(feat, tuple) else len(
                feat.size())
            assert feat_dim in [
                5, 2
            ], ('Got feature of unknown architecture, '
                'only 3D-CNN-like ([N, in_channels, T, H, W]), and '
                'transformer-like ([N, in_channels]) features are supported.')
            if feat_dim == 5:  # 3D-CNN architecture
                # perform spatio-temporal pooling
                avg_pool = nn.AdaptiveAvgPool3d(1)
                if isinstance(feat, tuple):
                    feat = [avg_pool(x) for x in feat]
                    # concat them
                    feat = torch.cat(feat, axis=1)
                else:
                    feat = avg_pool(feat)
                # squeeze dimensions
                feat = feat.reshape((batches, num_segs, -1))
                # temporal average pooling
                feat = feat.mean(axis=1)
            return feat

        # should have cls_head if not extracting features
        assert self.with_cls_head
        cls_score = self.cls_head(feat, domains=domains)
        return self.get_prob(cls_score)
