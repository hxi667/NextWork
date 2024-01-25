import torch.nn as nn

# loss between student and teacher
class betweenLoss(nn.Module):
    def __init__(self, gamma=[1, 1, 1, 1, 1, 1], loss=nn.L1Loss()):
        super(betweenLoss, self).__init__()
        self.gamma = gamma
        self.loss = loss

    def forward(self, outputs, targets):
        assert len(outputs)
        assert len(outputs) == len(targets)
        res = sum([self.gamma[i] * self.loss(outputs[i], targets[i]) for i in range(len(outputs))])
        return res