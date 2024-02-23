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

        # teacher
        self.teacher_conv1 = nn.Conv1d(256, 128, kernel_size=3, padding=1)
        self.teacher_conv2 = nn.Conv1d(128, 64, kernel_size=2, padding=1)
        self.teacher_relu = nn.ReLU()
        self.teacher_maxpool = nn.MaxPool1d(kernel_size=2, stride=2)
        self.teacher_fc = nn.Linear(64*16, 74*16)

    def forward(self, inputs):
        ipts, labs, _, _, seqL = inputs
        sils = ipts[0]  # [n, s, h, w]
        if len(sils.size()) == 4:
            sils = sils.unsqueeze(1)

        del ipts
        outs = self.set_block1(sils)
        gl = self.set_pooling(outs, seqL, options={"dim": 2})[0]
        gl = self.gl_block2(gl)

        outs = self.set_block2(outs)
        gl = gl + self.set_pooling(outs, seqL, options={"dim": 2})[0]
        gl = self.gl_block3(gl)

        outs = self.set_block3(outs)
        outs = self.set_pooling(outs, seqL, options={"dim": 2})[0]
        gl = gl + outs

        # Horizontal Pooling Matching, HPM
        feature1 = self.HPP(outs)  # [n, c, p]
        feature2 = self.HPP(gl)  # [n, c, p]
        feature = torch.cat([feature1, feature2], -1)  # [n, c, p]
        embs = self.Head(feature)

        # teacher output
        between = self.teacher_relu(self.teacher_conv1(embs))
        between = self.teacher_maxpool(between)
        between = self.teacher_relu(self.teacher_conv2(between))
        between = self.teacher_maxpool(between)
        between = self.teacher_fc(between.view(between.size(0), -1))
        between = between.view(between.size(0), 74, 16)

        
        n, _, s, h, w = sils.size()
        retval = {
            'training_feat': {
                'triplet': {'embeddings': embs, 'labels': labs} # embs: torch.Size([16, 256, 62])
            },
            
            'between_feat': [between], # between: [torch.Size([16, 74, 16])]
            
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
