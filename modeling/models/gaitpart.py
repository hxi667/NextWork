import torch
import torch.nn as nn
from ..modules import SetBlockWrapper, HorizontalPoolingPyramid, PackSequenceWrapper, SeparateFCs
from utils.common import clones
from utils.common import get_valid_args, is_list, is_dict, get_attr_from
from .. import backbones

class BasicConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, **kwargs):
        super(BasicConv1d, self).__init__()
        self.conv = nn.Conv1d(in_channels, out_channels,
                              kernel_size, bias=False, **kwargs)

    def forward(self, x):
        ret = self.conv(x)
        return ret


class TemporalFeatureAggregator(nn.Module):
    def __init__(self, in_channels, squeeze=4, parts_num=16):
        super(TemporalFeatureAggregator, self).__init__()
        hidden_dim = int(in_channels // squeeze)
        self.parts_num = parts_num

        # MTB1
        conv3x1 = nn.Sequential(
            BasicConv1d(in_channels, hidden_dim, 3, padding=1),
            nn.LeakyReLU(inplace=True),
            BasicConv1d(hidden_dim, in_channels, 1))
        self.conv1d3x1 = clones(conv3x1, parts_num)
        self.avg_pool3x1 = nn.AvgPool1d(3, stride=1, padding=1)
        self.max_pool3x1 = nn.MaxPool1d(3, stride=1, padding=1)

        # MTB1
        conv3x3 = nn.Sequential(
            BasicConv1d(in_channels, hidden_dim, 3, padding=1),
            nn.LeakyReLU(inplace=True),
            BasicConv1d(hidden_dim, in_channels, 3, padding=1))
        self.conv1d3x3 = clones(conv3x3, parts_num)
        self.avg_pool3x3 = nn.AvgPool1d(5, stride=1, padding=2)
        self.max_pool3x3 = nn.MaxPool1d(5, stride=1, padding=2)

        # Temporal Pooling, TP
        self.TP = torch.max

    def forward(self, x):
        """
          Input:  x,   [n, c, s, p]
          Output: ret, [n, c, p]
        """
        n, c, s, p = x.size()
        x = x.permute(3, 0, 1, 2).contiguous()  # [p, n, c, s]
        feature = x.split(1, 0)  # [[1, n, c, s], ...]
        x = x.view(-1, c, s)

        # MTB1: ConvNet1d & Sigmoid
        logits3x1 = torch.cat([conv(_.squeeze(0)).unsqueeze(0)
                               for conv, _ in zip(self.conv1d3x1, feature)], 0)
        scores3x1 = torch.sigmoid(logits3x1)
        # MTB1: Template Function
        feature3x1 = self.avg_pool3x1(x) + self.max_pool3x1(x)
        feature3x1 = feature3x1.view(p, n, c, s)
        feature3x1 = feature3x1 * scores3x1

        # MTB2: ConvNet1d & Sigmoid
        logits3x3 = torch.cat([conv(_.squeeze(0)).unsqueeze(0)
                               for conv, _ in zip(self.conv1d3x3, feature)], 0)
        scores3x3 = torch.sigmoid(logits3x3)
        # MTB2: Template Function
        feature3x3 = self.avg_pool3x3(x) + self.max_pool3x3(x)
        feature3x3 = feature3x3.view(p, n, c, s)
        feature3x3 = feature3x3 * scores3x3

        # Temporal Pooling
        ret = self.TP(feature3x1 + feature3x3, dim=-1)[0]  # [p, n, c]
        ret = ret.permute(1, 2, 0).contiguous()  # [n, p, c]
        return ret


class GaitPart(nn.Module):
    def __init__(self, model_cfg):
        super(GaitPart, self).__init__()
        """
            GaitPart: Temporal Part-based Model for Gait Recognition
            Paper:    https://openaccess.thecvf.com/content_CVPR_2020/papers/Fan_GaitPart_Temporal_Part-Based_Model_for_Gait_Recognition_CVPR_2020_paper.pdf
            Github:   https://github.com/ChaoFan96/GaitPart
        """
        self.restore_hint = 0
        self.load_ckpt_strict = True

        self.model_cfg = model_cfg

        self.Backbone = self.get_backbone( self.model_cfg['backbone_cfg'])
        head_cfg =  self.model_cfg['SeparateFCs']
        self.Head = SeparateFCs(** self.model_cfg['SeparateFCs'])
        self.Backbone = SetBlockWrapper(self.Backbone)
        self.HPP = SetBlockWrapper(
            HorizontalPoolingPyramid(bin_num= self.model_cfg['bin_num']))
        self.TFA = PackSequenceWrapper(TemporalFeatureAggregator(
            in_channels=head_cfg['in_channels'], parts_num=head_cfg['parts_num']))
        
        # teacher 
        self.teacher_conv1 = nn.Conv1d(128, 74, kernel_size=5, padding=2)
        self.teacher_relu = nn.ReLU()
        self.teacher_maxpool = nn.MaxPool1d(kernel_size=1, stride=1)

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

        del ipts
        # sils: torch.Size([batch, 1, 30, 64, 44])
        out = self.Backbone(sils)  # [n, c, s, h, w] # out: torch.Size([batch, 128, 30, 16, 11])
        self._add_feature(out, feature_maps, 0)     

        out = self.HPP(out)  # [n, c, s, p] # out: torch.Size([batch, 128, 30, 16])
        self._add_feature(out, feature_maps, 1)     

        out = self.TFA(out, seqL)  # [n, c, p] # out: torch.Size([batch, 128, 16])
        self._add_feature(out, feature_maps, 2)
 
        embs = self.Head(out)  # [n, c, p] # out: torch.Size([batch, 128, 16])
        self._add_feature(embs, feature_maps, 3)

        # teacher output
        between = self.teacher_relu(self.teacher_conv1(embs))
        between = self.teacher_maxpool(between) # between: [torch.Size([batch, 74, 16])]
        self._add_feature(between, feature_maps, 4)

        n, _, s, h, w = sils.size()
        retval = {
            'training_feat': {
                'triplet': {'embeddings': embs, 'labels': labs} # embs: torch.Size([batch, 128, 16])
            },

            'between_feat': feature_maps, # between: [torch.Size([batch, 74, 16])]

            'visual_summary': {
                'image/sils': sils.view(n*s, 1, h, w)
            },
            'inference_feat': {
                'embeddings': embs
            }
        }
        return retval


def gaitPart(model_cfg):
    gaitPart_cfg = {
        'backbone_cfg': {
            'type': 'Plain',
            'in_channels': 1,
            'layers_cfg': ["BC-32", "BC-32", "M", "FC-64-2", "FC-64-2", "M", "FC-128-3", "FC-128-3"],
        },
        'SeparateFCs': {
            'in_channels': 128,
            'out_channels': 128,
            'parts_num': 16,
        },
        'bin_num': [16]
    }

    keys_to_merge = ['fc_out', 'pool_out', 'out_dims'] 

    merged_dict = {key: model_cfg[key] for key in keys_to_merge if key in model_cfg}
    merged_dict.update(gaitPart_cfg)  

    return GaitPart(merged_dict)

