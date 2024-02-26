import torch
import torch.nn as nn
from ..modules import SetBlockWrapper, HorizontalPoolingPyramid, PackSequenceWrapper, SeparateFCs, SeparateBNNecks

from einops import rearrange
from utils.common import get_valid_args, is_list, is_dict, get_attr_from
from .. import backbones

class Baseline(nn.Module):

    def __init__(self, model_cfg):
        super(Baseline, self).__init__()

        self.restore_hint = 0
        self.load_ckpt_strict = True

        self.model_cfg = model_cfg

        self.Backbone = self.get_backbone(self.model_cfg['backbone_cfg'])
        self.Backbone = SetBlockWrapper(self.Backbone)
        self.FCs = SeparateFCs(**self.model_cfg['SeparateFCs'])
        self.BNNecks = SeparateBNNecks(**self.model_cfg['SeparateBNNecks'])
        self.TP = PackSequenceWrapper(torch.max)
        self.HPP = HorizontalPoolingPyramid(bin_num=self.model_cfg['bin_num'])

        # add_feature' parameters 
        self.pool_out = self.model_cfg['pool_out'] # or avg
        self.fc_out = self.model_cfg['fc_out']
        self.out_dims = self.model_cfg['out_dims'][-5:]
        if self.fc_out:
            self.fc_layers = self._make_fc()

    # 获取 backbone model
    def get_backbone(self, backbone_cfg):
        """Get the backbone of the model."""
        # dict 类型
        if is_dict(backbone_cfg):
            # 从 backbones 包中获取名为 backbone_cfg['type'] 的 Backbone 类 
            Backbone = get_attr_from([backbones], backbone_cfg['type'])
            # 验证 Backbone 类的参数
            valid_args = get_valid_args(Backbone, backbone_cfg, ['type'])
            # 返回此 Backbone 类实例 
            return Backbone(**valid_args)
        # list 类型
        if is_list(backbone_cfg):
            # 递归调用 get_backbone(), 并将结果封装成一个 nn.ModuleList
            Backbone = nn.ModuleList([self.get_backbone(cfg)
                                      for cfg in backbone_cfg])
            return Backbone
        raise ValueError(
            "Error type for -Backbone-Cfg-, supported: (A list of) dict.")  

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

    def forward(self, inputs):
        feature_maps = []

        ipts, labs, _, _, seqL = inputs

        sils = ipts[0]
        if len(sils.size()) == 4:
            sils = sils.unsqueeze(1)
        else:
            sils = rearrange(sils, 'n s c h w -> n c s h w')

        del ipts
        # sils: torch.Size([batch, 1, 30, 64, 44])
        outs = self.Backbone(sils)  # [n, c, s, h, w] # outs: torch.Size([batch, 512, 30, 16, 11])
        self._add_feature(outs, feature_maps, 0)

        # Temporal Pooling, TP
        outs = self.TP(outs, seqL, options={"dim": 2})[0]  # [n, c, h, w] # outs: torch.Size([batch, 512, 16, 11])
        self._add_feature(outs, feature_maps, 1)

        # Horizontal Pooling Matching, HPM
        feat = self.HPP(outs)  # [n, c, p] # feat: torch.Size([batch, 512, 16])
        self._add_feature(feat, feature_maps, 2)

        embed_1 = self.FCs(feat)  # [n, c, p] # embed_1: torch.Size([batch, 256, 16])
        self._add_feature(embed_1, feature_maps, 3)

        embed_2, logits = self.BNNecks(embed_1)  # [n, c, p] # embed_2: torch.Size([batch, 256, 16]), logits: torch.Size([batch, 74, 16])
        self._add_feature(embed_2, feature_maps, 4)
        
        embed = embed_1 # embed:

        retval = {
            'training_feat': {
                'triplet': {'embeddings': embed_1, 'labels': labs}, # embed_1:torch.Size([batch, 256, 16]),  labs:torch.Size([batch])
                'softmax': {'logits': logits, 'labels': labs} # logits: torch.Size([batch, 74, 16])
            },

            'between_feat': feature_maps, 

            'visual_summary': {
                'image/sils': rearrange(sils,'n c s h w -> (n s) c h w')
            },

            'inference_feat': {
                'embeddings': embed
            }
        }
        return retval
    

def baseline_ResNet9(model_cfg):
    baseline_ResNet9_cfg = {
        'backbone_cfg': {
            'type': "ResNet9",
            'block': "BasicBlock",
            'channels': [64, 128, 256, 512],
            'layers': [1, 1, 1, 1],
            'strides': [1, 2, 2, 1],
            'maxpool': False,
        },
        'SeparateFCs': {
            'in_channels': 512,
            'out_channels': 256,
            'parts_num': 16,
        },
        'SeparateBNNecks': {
            'class_num': 74,
            'in_channels': 256,
            'parts_num': 16,
        },
        'bin_num': [16]
    }

    keys_to_merge = ['fc_out', 'pool_out', 'out_dims'] 

    merged_dict = {key: model_cfg[key] for key in keys_to_merge if key in model_cfg}
    merged_dict.update(baseline_ResNet9_cfg)  

    return Baseline(merged_dict)

