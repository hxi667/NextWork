import torch
import torch.nn as nn
from torch.cuda.amp import autocast

class discriminatorLoss(nn.Module):
    def __init__(self, models, eta=[1, 1, 1, 1, 1], enable_float16=False, loss=nn.BCEWithLogitsLoss(), ):
        super(discriminatorLoss, self).__init__()
        self.models = models
        self.eta = eta
        self.loss = loss
        self.enable_float16 = enable_float16

    def forward(self, outputs, targets):
        inputs = [torch.cat((i,j),0) for i, j in zip(outputs, targets)]
        batch_size = inputs[0].size(0)
        target = torch.FloatTensor([[1, 0] for _ in range(batch_size//2)] + [[0, 1] for _ in range(batch_size//2)])
        target = target.to(inputs[0].device)
        with autocast(enabled=self.enable_float16):
            outputs = self.models(inputs)
            res = sum([self.eta[i] * self.loss(output, target) for i, output in enumerate(outputs)])
        return res
    
    
# 未添加鉴别器时的假loss
class discriminatorFakeLoss(nn.Module):
    def forward(self, outputs, targets):
        res = (0*outputs[0]).sum()
        return res