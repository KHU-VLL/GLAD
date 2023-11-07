from ..builder import LOSSES
from .base import BaseWeightedLoss

import torch


@LOSSES.register_module()
class DANNLoss(BaseWeightedLoss):
    def __init__(self, num_classes, weight_loss_domain=.5):
        super().__init__()
        self.num_classes = num_classes
        self.loss_cls = DANNClassifierLoss(1.-weight_loss_domain)
        self.loss_domain = DANNDomainLoss(weight_loss_domain)
    
    def _forward(self, cls_score:list, label, domains=None, **kwargs):
        cls_score, domain_score = cls_score
        loss_cls = self.loss_cls(cls_score, label, domains, **kwargs)
        loss_domain = self.loss_domain(domain_score, None, domains, **kwargs)
        return {
            'loss_cls': loss_cls,
            'loss_domain': loss_domain,
        }


@LOSSES.register_module()
class DANNClassifierLoss(BaseWeightedLoss):

    def __init__(self, num_classes):
        super().__init__()
        self.num_classes = num_classes
        self.cls_loss = torch.nn.CrossEntropyLoss()
        
    def _forward(self, cls_score, label, domains=None, **kwargs):
        """
        Args:
            cls_score: The output of the classification head of source-domain videos
        
        Returns:
            torch.Tensor: Computed loss.
        """
        loss_cls = self.cls_loss(cls_score, label)
        return loss_cls


@LOSSES.register_module()
class DANNDomainLoss(BaseWeightedLoss):

    def __init__(self, loss_weight=1.):
        super().__init__(loss_weight)
        self.domain_loss = torch.nn.BCEWithLogitsLoss()
    
    def _forward(self, cls_score, label, domains=None, **kwargs):
        # can't edit the argument's name, so edited here
        domain_score = cls_score
        domain_label = label   # actually not being used

        source_idx = torch.from_numpy(domains == 'source')
        target_idx = torch.logical_not(source_idx)
        assert source_idx.sum() == target_idx.sum()

        loss_domain = self.domain_loss(domain_score, source_idx.type(torch.float32).to('cuda').unsqueeze(1))
        return loss_domain
