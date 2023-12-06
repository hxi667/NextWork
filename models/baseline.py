import torch
import torch.nn as nn
from models.modules import SetBlockWrapper, HorizontalPoolingPyramid, PackSequenceWrapper, SeparateFCs, SeparateBNNecks

from einops import rearrange
from utils import get_valid_args, is_list, is_dict, get_attr_from
from . import backbones

class Baseline(nn.Module):

    def __init__(self, model_cfg):
        super(Baseline, self).__init__()
        self.Backbone = self.get_backbone(model_cfg['backbone_cfg'])
        self.Backbone = SetBlockWrapper(self.Backbone)
        self.FCs = SeparateFCs(**model_cfg['SeparateFCs'])
        self.BNNecks = SeparateBNNecks(**model_cfg['SeparateBNNecks'])
        self.TP = PackSequenceWrapper(torch.max)
        self.HPP = HorizontalPoolingPyramid(bin_num=model_cfg['bin_num'])


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

    def forward(self, inputs):
        ipts, labs, _, _, seqL = inputs

        sils = ipts[0]
        if len(sils.size()) == 4:
            sils = sils.unsqueeze(1)
        else:
            sils = rearrange(sils, 'n s c h w -> n c s h w')

        del ipts
        outs = self.Backbone(sils)  # [n, c, s, h, w]

        # Temporal Pooling, TP
        outs = self.TP(outs, seqL, options={"dim": 2})[0]  # [n, c, h, w]
        # Horizontal Pooling Matching, HPM
        feat = self.HPP(outs)  # [n, c, p]

        embed_1 = self.FCs(feat)  # [n, c, p]
        embed_2, logits = self.BNNecks(embed_1)  # [n, c, p]
        embed = embed_1

        retval = {
            'training_feat': {
                'triplet': {'embeddings': embed_1, 'labels': labs},
                'softmax': {'logits': logits, 'labels': labs}
            },
            'visual_summary': {
                'image/sils': rearrange(sils,'n c s h w -> (n s) c h w')
            },
            'inference_feat': {
                'embeddings': embed
            }
        }
        return retval
    

def baseline_ResNet9():
    model_cfg = {
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
        'bin_num': 16
    }
    return Baseline(model_cfg)