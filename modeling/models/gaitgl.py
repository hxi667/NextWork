import torch
import torch.nn as nn
import torch.nn.functional as F

# from models.base_model import BaseModel
from ..modules import SeparateFCs, BasicConv3d, PackSequenceWrapper, SeparateBNNecks


class GLConv(nn.Module):
    def __init__(self, in_channels, out_channels, halving, fm_sign=False, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1), bias=False, **kwargs):
        super(GLConv, self).__init__()
        self.halving = halving
        self.fm_sign = fm_sign
        self.global_conv3d = BasicConv3d(
            in_channels, out_channels, kernel_size, stride, padding, bias, **kwargs)
        self.local_conv3d = BasicConv3d(
            in_channels, out_channels, kernel_size, stride, padding, bias, **kwargs)

    def forward(self, x):
        '''
            x: [n, c, s, h, w]
        '''
        gob_feat = self.global_conv3d(x)
        if self.halving == 0:
            lcl_feat = self.local_conv3d(x)
        else:
            h = x.size(3)
            split_size = int(h // 2**self.halving)
            lcl_feat = x.split(split_size, 3)
            lcl_feat = torch.cat([self.local_conv3d(_) for _ in lcl_feat], 3)

        if not self.fm_sign:
            feat = F.leaky_relu(gob_feat) + F.leaky_relu(lcl_feat)
        else:
            feat = F.leaky_relu(torch.cat([gob_feat, lcl_feat], dim=3))
        return feat


class GeMHPP(nn.Module):
    def __init__(self, bin_num=[64], p=6.5, eps=1.0e-6):
        super(GeMHPP, self).__init__()
        self.bin_num = bin_num
        self.p = nn.Parameter(
            torch.ones(1)*p)
        self.eps = eps

    def gem(self, ipts):
        return F.avg_pool2d(ipts.clamp(min=self.eps).pow(self.p), (1, ipts.size(-1))).pow(1. / self.p)

    def forward(self, x):
        """
            x  : [n, c, h, w]
            ret: [n, c, p] 
        """
        n, c = x.size()[:2]
        features = []
        for b in self.bin_num:
            z = x.view(n, c, b, -1)
            z = self.gem(z).squeeze(-1)
            features.append(z)
        return torch.cat(features, -1)


class GaitGL(nn.Module):
    """
        GaitGL: Gait Recognition via Effective Global-Local Feature Representation and Local Temporal Aggregation
        Arxiv : https://arxiv.org/pdf/2011.01461.pdf
    """

    def __init__(self, model_cfg, dataset_name="CASIA-B"):
        super(GaitGL, self).__init__()

        self.restore_hint = 0
        self.load_ckpt_strict = True
        
        self.model_cfg = model_cfg
        
        in_c = self.model_cfg['channels']
        class_num = self.model_cfg['class_num']
        
        
        if dataset_name in ['OUMVLP', 'GREW']:
            # For OUMVLP and GREW
            self.conv3d = nn.Sequential(
                BasicConv3d(1, in_c[0], kernel_size=(3, 3, 3),
                            stride=(1, 1, 1), padding=(1, 1, 1)),
                nn.LeakyReLU(inplace=True),
                BasicConv3d(in_c[0], in_c[0], kernel_size=(3, 3, 3),
                            stride=(1, 1, 1), padding=(1, 1, 1)),
                nn.LeakyReLU(inplace=True),
            )
            self.LTA = nn.Sequential(
                BasicConv3d(in_c[0], in_c[0], kernel_size=(
                    3, 1, 1), stride=(3, 1, 1), padding=(0, 0, 0)),
                nn.LeakyReLU(inplace=True)
            )

            self.GLConvA0 = nn.Sequential(
                GLConv(in_c[0], in_c[1], halving=1, fm_sign=False, kernel_size=(
                    3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1)),
                GLConv(in_c[1], in_c[1], halving=1, fm_sign=False, kernel_size=(
                    3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1)),
            )
            self.MaxPool0 = nn.MaxPool3d(
                kernel_size=(1, 2, 2), stride=(1, 2, 2))

            self.GLConvA1 = nn.Sequential(
                GLConv(in_c[1], in_c[2], halving=1, fm_sign=False, kernel_size=(
                    3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1)),
                GLConv(in_c[2], in_c[2], halving=1, fm_sign=False, kernel_size=(
                    3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1)),
            )
            self.GLConvB2 = nn.Sequential(
                GLConv(in_c[2], in_c[3], halving=1, fm_sign=False,  kernel_size=(
                    3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1)),
                GLConv(in_c[3], in_c[3], halving=1, fm_sign=True,  kernel_size=(
                    3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1)),
            )
        else:
            # For CASIA-B or other unstated datasets.
            self.conv3d = nn.Sequential(
                BasicConv3d(1, in_c[0], kernel_size=(3, 3, 3),
                            stride=(1, 1, 1), padding=(1, 1, 1)),
                nn.LeakyReLU(inplace=True)
            )
            self.LTA = nn.Sequential(
                BasicConv3d(in_c[0], in_c[0], kernel_size=(
                    3, 1, 1), stride=(3, 1, 1), padding=(0, 0, 0)),
                nn.LeakyReLU(inplace=True)
            )

            self.GLConvA0 = GLConv(in_c[0], in_c[1], halving=3, fm_sign=False, kernel_size=(
                3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))
            self.MaxPool0 = nn.MaxPool3d(
                kernel_size=(1, 2, 2), stride=(1, 2, 2))

            self.GLConvA1 = GLConv(in_c[1], in_c[2], halving=3, fm_sign=False, kernel_size=(
                3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))
            self.GLConvB2 = GLConv(in_c[2], in_c[2], halving=3, fm_sign=True,  kernel_size=(
                3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))

        self.TP = PackSequenceWrapper(torch.max)
        self.HPP = GeMHPP()

        self.Head0 = SeparateFCs(64, in_c[-1], in_c[-1])

        if 'SeparateBNNecks' in self.model_cfg.keys():
            self.BNNecks = SeparateBNNecks(**self.model_cfg['SeparateBNNecks'])
            self.Bn_head = False
        else:
            self.Bn = nn.BatchNorm1d(in_c[-1])
            self.Head1 = SeparateFCs(64, in_c[-1], class_num)
            self.Bn_head = True

        # # teacher
        # self.teacher_linear = nn.Linear(64, 16) ###
            
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

    def forward(self, inputs, training=True):
        feature_maps = []

        ipts, labs, _, _, seqL = inputs
        seqL = None if not training else seqL
        if not training and len(labs) != 1:
            raise ValueError(
                'The input size of each GPU must be 1 in testing mode, but got {}!'.format(len(labs)))
        
        sils = ipts[0].unsqueeze(1)
        del ipts
        n, _, s, h, w = sils.size()
        if s < 3:
            repeat = 3 if s == 1 else 2
            sils = sils.repeat(1, 1, repeat, 1, 1)

        # sils: torch.Size([batch, 1, 30, 64, 44])
        outs = self.conv3d(sils) # outs：torch.Size([batch, 32, 30, 64, 44])
        outs = self.LTA(outs) # outs：torch.Size([batch, 32, 10, 64, 44])
        self._add_feature(outs, feature_maps, 0)

        outs = self.GLConvA0(outs) # outs：torch.Size([batch, 64, 10, 64, 44])
        outs = self.MaxPool0(outs) # outs：torch.Size([batch, 64, 10, 32, 22])
        self._add_feature(outs, feature_maps, 1)

        outs = self.GLConvA1(outs) # outs：torch.Size([batch, 128, 10, 32, 22])
        outs = self.GLConvB2(outs)  # [n, c, s, h, w] # outs：torch.Size([batch, 128, 10, 64, 22])
        self._add_feature(outs, feature_maps, 2)

        outs = self.TP(outs, seqL=seqL, options={"dim": 2})[0]  # [n, c, h, w] # outs：torch.Size([batch, 128, 64, 22])
        outs = self.HPP(outs)  # [n, c, p] # outs：torch.Size([batch, 128, 64])
        self._add_feature(outs, feature_maps, 3)

        gait = self.Head0(outs)  # [n, c, p] # gait.Size([batch, 128, 64])

        if self.Bn_head:  # Original GaitGL Head
            bnft = self.Bn(gait)  # [n, c, p] # bnft：torch.Size([batch, 128, 64])
            logi = self.Head1(bnft)  # [n, c, p] # logi: torch.Size([batch, 74, 64])
            embed = bnft
            self._add_feature(embed, feature_maps, 4)
        
        else:  # BNNechk as Head
            bnft, logi = self.BNNecks(gait)  # [n, c, p] 
            embed = gait 

        # # teacher output
        # between = self.teacher_linear(logi) # torch.Size([batch, 74, 16])

        n, _, s, h, w = sils.size()
        retval = {
            'training_feat': {
                'triplet': {'embeddings': embed, 'labels': labs}, # embed: torch.Size([batch, 128, 64])
                'softmax': {'logits': logi, 'labels': labs} # logi: torch.Size([batch, 74, 64])
            },

            'between_feat': feature_maps, # between: [torch.Size([batch, 74, 16])]
            
            'visual_summary': {
                'image/sils': sils.view(n*s, 1, h, w)
            },

            'inference_feat': {
                'embeddings': embed
            }
        }
        return retval


def gaitGL(model_cfg):

    gaitGL_cfg = {
        'channels': [32, 64, 128],
        'class_num': 74
    }

    keys_to_merge = ['fc_out', 'pool_out', 'out_layer', 'out_dims'] 

    merged_dict = {key: model_cfg[key] for key in keys_to_merge if key in model_cfg}
    merged_dict.update(gaitGL_cfg)  

    return GaitGL(merged_dict, "CASIA-B")