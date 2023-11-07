from collections import Iterable

import numpy as np
import torch
import torch.nn as nn

from einops import rearrange

from ..builder import RECOGNIZERS
from .base import BaseRecognizer


@RECOGNIZERS.register_module()
class BaseDARecognizer(BaseRecognizer):
    def train_step(self, data_batches, domains, optimizer=None, **kwargs):
        imgs_source, imgs_target = data_batches[0]['imgs'], data_batches[1]['imgs']
        # imgs = [imgs_source, imgs_target]
        labels_source, labels_target = data_batches[0]['label'], data_batches[1]['label']  # D x [B, M], M=1 if single-labeled
        domains = np.array([
            [domain]*data_batch['imgs'].shape[0] for domain, data_batch in zip(domains, data_batches)
        ]).reshape(-1)  # [D x B]

        # shuffle the batch here
        B = imgs_source.shape[0]
        device = imgs_source.device
        indices = torch.randperm(B)
        imgs = [imgs_source[indices], imgs_target[indices]]
        labels = torch.cat([labels_source[indices], labels_target[indices]]).to(device)
        # domains = domains[indices]  # no need to shuffle

        losses = self(imgs, labels, domains=domains, return_loss=True, **kwargs)

        loss, log_vars = self._parse_losses(losses)

        outputs = dict(
            loss=loss,
            log_vars=log_vars,
            num_samples=2*B)

        return outputs

    def forward(self, imgs, label=None, domains=None, return_loss=True, **kwargs):
        """Define the computation performed at every call."""
        if kwargs.get('gradcam', False):
            del kwargs['gradcam']
            return self.forward_gradcam(imgs, **kwargs)
        if return_loss:
            if label is None:
                raise ValueError('Label should not be None.')
            if self.blending is not None:
                imgs, label = self.blending(imgs, label)
            return self.forward_train(imgs, label, domains=domains, **kwargs)
        return self.forward_test(imgs, domains, **kwargs)

    def forward_neck(self, outs, labels, domains=None, **kwargs):
        losses_aux = dict()
        vertebrae = self.neck
        if not isinstance(vertebrae, Iterable):
            vertebrae = [vertebrae]
        for vertebra in vertebrae:
            outs, loss_aux = vertebra(outs, labels, domains, **kwargs)
            if loss_aux is not None:
                losses_aux.update(loss_aux)
        return outs, losses_aux

    def forward_test(self, imgs, domains=None):  # val_step도 수정할 거면 자식으로 내려야 됨
        if self.test_cfg.get('fcn_test', False):
            # If specified, spatially fully-convolutional testing is performed
            assert not self.feature_extraction
            assert self.with_cls_head
            return self._do_fcn_test(imgs).cpu().numpy()
        return self._do_test(imgs, domains).cpu().numpy()

    def forward_gradcam(self, imgs, **kwargs):
        assert self.with_cls_head
        return self._do_test(imgs, **kwargs)
