import torch.nn as nn
import torch.optim as optim


class MaskedLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, output, target, mask_mat):
        residual = ((output.flatten() - target.flatten()) ** 2) * mask_mat.flatten()
        loss = residual.sum() / mask_mat.sum()
        return loss