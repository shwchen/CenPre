import torch
import torch.nn as nn
import torch.nn.functional as f
import numpy as np


class ContrastiveAlignmentLoss(nn.Module):
    def __init__(self, fixed_scale=None, reduction='mean', device=None):
        super(ContrastiveAlignmentLoss, self).__init__()
        self.device = device
        if fixed_scale is not None:
            self.logit_scale = fixed_scale.to(self.device)
        else:
            self.logit_scale = torch.exp(torch.tensor(np.log(1 / 0.07))).to(self.device)
        self.reduction = reduction

    def forward(self, features1, features2):
        logits = self.logit_scale * features1 @ features2.t()
        labels = torch.arange(logits.size(0), device=self.device)

        loss_i = f.cross_entropy(logits, labels, reduction=self.reduction)
        loss_t = f.cross_entropy(logits.t(), labels, reduction=self.reduction)

        if self.reduction == 'mean':
            return (loss_i + loss_t) / 2
        elif self.reduction == 'sum':
            return loss_i + loss_t
        else:
            return loss_i, loss_t
