import torch
import copy
import torch.nn as nn

from ..modules import SeparateFCs, BasicConv2d, SetBlockWrapper, HorizontalPoolingPyramid, PackSequenceWrapper


class GaitSet(nn.Module):
    """
        GaitSet: Regarding Gait as a Set for Cross-View Gait Recognition
        Arxiv:  https://arxiv.org/abs/1811.06186
        Github: https://github.com/AbnerHqC/GaitSet
    """

    def __init__(self, model_cfg):
        super(GaitSet, self).__init__()

        self.restore_hint = 0
        self.load_ckpt_strict = True

        self.model_cfg = model_cfg

        in_c = self.model_cfg['in_channels']
        self.set_block1 = nn.Sequential(BasicConv2d(in_c[0], in_c[1], 5, 1, 2),
                                        nn.LeakyReLU(inplace=True),
                                        BasicConv2d(in_c[1], in_c[1], 3, 1, 1),
                                        nn.LeakyReLU(inplace=True),
                                        nn.MaxPool2d(kernel_size=2, stride=2))

        self.set_block2 = nn.Sequential(BasicConv2d(in_c[1], in_c[2], 3, 1, 1),
                                        nn.LeakyReLU(inplace=True),
                                        BasicConv2d(in_c[2], in_c[2], 3, 1, 1),
                                        nn.LeakyReLU(inplace=True),
                                        nn.MaxPool2d(kernel_size=2, stride=2))

        self.set_block3 = nn.Sequential(BasicConv2d(in_c[2], in_c[3], 3, 1, 1),
                                        nn.LeakyReLU(inplace=True),
                                        BasicConv2d(in_c[3], in_c[3], 3, 1, 1),
                                        nn.LeakyReLU(inplace=True))

        self.gl_block2 = copy.deepcopy(self.set_block2)
        self.gl_block3 = copy.deepcopy(self.set_block3)

        self.set_block1 = SetBlockWrapper(self.set_block1)
        self.set_block2 = SetBlockWrapper(self.set_block2)
        self.set_block3 = SetBlockWrapper(self.set_block3)

        self.set_pooling = PackSequenceWrapper(torch.max)

        self.Head = SeparateFCs(**self.model_cfg['SeparateFCs'])

        self.HPP = HorizontalPoolingPyramid(bin_num=self.model_cfg['bin_num'])

        # # teacher
        # self.teacher_conv1 = nn.Conv1d(256, 128, kernel_size=3, padding=1)
        # self.teacher_conv2 = nn.Conv1d(128, 64, kernel_size=2, padding=1)
        # self.teacher_relu = nn.ReLU()
        # self.teacher_maxpool = nn.MaxPool1d(kernel_size=2, stride=2)
        # self.teacher_fc = nn.Linear(64*16, 74*16)

        # add_feature' parameters 
        self.pool_out = self.model_cfg['pool_out'] # or avg
        self.fc_out = self.model_cfg['fc_out']
        self.out_dims = self.model_cfg['out_dims'][-5:]
        if self.fc_out:
            self.fc_layers = self._make_fc()

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
        sils = ipts[0]  # [n, s, h, w]
        if len(sils.size()) == 4:
            sils = sils.unsqueeze(1)

        del ipts
        # sils: torch.Size([batch, 1, 30, 64, 44])
        outs = self.set_block1(sils) # outs: torch.Size([batch, 32, 30, 32, 22])
        gl = self.set_pooling(outs, seqL, options={"dim": 2})[0] # gl: torch.Size([batch, 32, 32, 22])
        gl = self.gl_block2(gl) # gl: torch.Size([batch, 64, 16, 11])
        self._add_feature(gl, feature_maps, 0)

        outs = self.set_block2(outs) # outs: torch.Size([batch, 64, 30, 16, 11])
        gl = gl + self.set_pooling(outs, seqL, options={"dim": 2})[0] # gl: torch.Size([batch, 64, 16, 11])
        gl = self.gl_block3(gl) # gl: torch.Size([batch, 128, 16, 11])
        self._add_feature(gl, feature_maps, 1)

        outs = self.set_block3(outs) # outs: torch.Size([batch, 128, 30, 16, 11])
        outs = self.set_pooling(outs, seqL, options={"dim": 2})[0] # outs: torch.Size([batch, 128, 16, 11])
        gl = gl + outs # gl: torch.Size([batch, 128, 16, 11])
        self._add_feature(gl, feature_maps, 2)        

        # Horizontal Pooling Matching, HPM
        feature1 = self.HPP(outs)  # [n, c, p] # feature1: torch.Size([batch, 128, 31])
        feature2 = self.HPP(gl)  # [n, c, p] # feayire2: torch.Size([batch, 128, 31])
        feature = torch.cat([feature1, feature2], -1)  # [n, c, p] # feature: torch.Size([batch, 128, 62])
        self._add_feature(feature, feature_maps, 3)

        embs = self.Head(feature) # embs: torch.Size([batch, 256, 62])
        self._add_feature(embs, feature_maps, 4)

        # # teacher output
        # between = self.teacher_relu(self.teacher_conv1(embs))
        # between = self.teacher_maxpool(between)
        # between = self.teacher_relu(self.teacher_conv2(between))
        # between = self.teacher_maxpool(between)
        # between = self.teacher_fc(between.view(between.size(0), -1))
        # between = between.view(between.size(0), 74, 16)

        
        n, _, s, h, w = sils.size()
        retval = {
            'training_feat': {
                'triplet': {'embeddings': embs, 'labels': labs} # embs: torch.Size([batch, 256, 62])
            },
            
            'between_feat': feature_maps,
            
            'visual_summary': {
                'image/sils': sils.view(n*s, 1, h, w)
            },

            'inference_feat': {
                'embeddings': embs
            }
        }
        return retval


def gaitSet(model_cfg):
    gaitSet_cfg = {
        'in_channels': [1, 32, 64, 128],
        'SeparateFCs': {
            'in_channels': 128,
            'out_channels': 256,
            'parts_num': 62,
        },
        'bin_num': [16, 8, 4, 2, 1]
    }

    keys_to_merge = ['fc_out', 'pool_out', 'out_layer', 'out_dims'] 

    merged_dict = {key: model_cfg[key] for key in keys_to_merge if key in model_cfg}
    merged_dict.update(gaitSet_cfg)  

    return GaitSet(merged_dict)
