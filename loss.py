import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


class MaskedLoss(nn.Module):
    '''
    TODO: structurally adapt from S. Wu et al
    '''
    def __init__(self):
        super(MaskedLoss, self).__init__()

    def forward(self, output, target, mask_mat):
        output = output.flatten()
        target = target.flatten()
        mask_mat = mask_mat.flatten()
        residual = (output - target) * mask_mat
        loss = residual.sum() / mask_mat.sum()
        return loss