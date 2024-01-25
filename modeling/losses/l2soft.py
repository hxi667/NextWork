import torch.nn.functional as F


def L2_soft(outputs, targets):
    softmax_outputs = F.softmax(outputs, dim=1)
    softmax_targets = F.softmax(targets, dim=1)

    return F.mse_loss(softmax_outputs, softmax_targets)