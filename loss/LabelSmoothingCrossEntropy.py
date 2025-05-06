import torch.nn as nn
import torch.nn.functional as f


class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self, smoothing=0.1):
        super(LabelSmoothingCrossEntropy, self).__init__()
        self.smoothing = smoothing

    def forward(self, pred, target):
        log_prob = f.log_softmax(pred, dim=-1)

        weight = pred.new_ones(pred.size()) * self.smoothing / (pred.size(-1) - 1.)
        weight.scatter_(-1, target.unsqueeze(-1), (1. - self.smoothing))

        loss = (-weight * log_prob).sum(dim=-1).mean()
        return loss
