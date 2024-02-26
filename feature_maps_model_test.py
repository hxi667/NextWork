import torch
import torch.nn as nn
import torch.nn.functional as F



class ResNet(nn.Module):
    def __init__(self):
        super(ResNet, self).__init__()

        self.pool_out = 'max' # or avg
        self.fc_out = True
        self.out_dims = [5000, 1000, 500, 200, 10][-5:]
        if self.fc_out:
            self.fc_layers = self._make_fc()

        self.linear = nn.Linear(512, 10)


    def _make_fc(self):
        if self.pool_out == "avg":
            layers = [
                nn.AdaptiveAvgPool1d(output) for output in self.out_dims
            ]
        elif self.pool_out == "max":
            layers = [
                nn.AdaptiveMaxPool1d(output) for output in self.out_dims
            ]
        return nn.Sequential(*layers)

    def _add_feature(self, x, feature_maps, fc_layer):
        if self.fc_out:
            out = self.fc_layers[fc_layer](x.view(x.size(0), x.size(1), -1))
            if self.pool_out == "max":
                out, _ = out.max(dim=1)
            else:
                out = out.mean(dim=1)
            feature_maps.append(torch.squeeze(out))
        else:
            feature_maps.append(x.view(x.size(0), -1))

            
    def forward(self, x):
        feature_maps = []

        # x1 = torch.randn(3, 64, 32, 32)
        x1 = torch.randn(3, 128, 10, 64,22)
        self._add_feature(x1, feature_maps, 0) 
        
        # x2 = torch.randn(3, 128, 16, 16)
        x2 = torch.randn(3, 128, 10, 64,22) 
        self._add_feature(x2, feature_maps, 1) 

        # x3 = torch.randn(3, 256, 8, 8)
        x3 = torch.randn(3, 256, 8, 8)
        self._add_feature(x3, feature_maps, 2) 

        # x4 = torch.randn(3, 512, 4, 4)
        x4 = torch.randn(3, 512, 4, 4)  
        self._add_feature(x4, feature_maps, 3) 

        out = F.avg_pool2d(x4, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out) # torch.Size([3, 10])

        feature_maps.append(out.view(out.size(0), -1))

        # [torch.Size([batch, 5000]), torch.Size([batch, 1000]), torch.Size([batch, 500]), torch.Size([batch, 200]), torch.Size([batch, 10])]
        return feature_maps

x = torch.randn(3, 10) 
net = ResNet()
feature_maps = net(x)
print("feature_maps[0].shape: ", feature_maps[0].shape)
print("feature_maps[1].shape: ", feature_maps[1].shape)
print("feature_maps[2].shape: ", feature_maps[2].shape)
print("feature_maps[3].shape: ", feature_maps[3].shape)
print("feature_maps[4].shape: ", feature_maps[4].shape)
