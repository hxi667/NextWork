import torch
import torch.nn as nn

class discriminatorLoss(nn.Module):
    def __init__(self, models, eta=[1, 1, 1, 1, 1], loss=nn.BCEWithLogitsLoss()):
        super(discriminatorLoss, self).__init__()
        self.models = models
        self.eta = eta
        self.loss = loss

    def forward(self, outputs, targets):
        inputs = [torch.cat((i,j),0) for i, j in zip(outputs, targets)]
        batch_size = inputs[0].size(0)
        target = torch.FloatTensor([[1, 0] for _ in range(batch_size//2)] + [[0, 1] for _ in range(batch_size//2)])
        target = target.to(inputs[0].device)
        outputs = self.models(inputs)
        res = sum([self.eta[i] * self.loss(output, target) for i, output in enumerate(outputs)])
        return res
    
    
# 未添加鉴别器时的假loss
class discriminatorFakeLoss(nn.Module):
    def forward(self, outputs, targets):
        res = (0*outputs[0]).sum()
        return res