import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
from models.vox.ResNetSE34V2_audio import MainModel as resnet34_audio
from models.vox.backbone_visual import resnet34 as resnet34_visual
from models.vox.fusion_modules import SumFusion, ConcatFusion, FiLM, GatedFusion


class AVClassifier(nn.Module):
    def __init__(self, args):
        super(AVClassifier, self).__init__()

        fusion = args.fusion_method
        if args.dataset == 'VGGSound':
            n_classes = 309
        elif args.dataset == 'KineticSound':
            n_classes = 31
        elif args.dataset == 'CREMAD':
            n_classes = 6
        elif args.dataset == 'AVE':
            n_classes = 28
        elif args.dataset == 'UCF101':
            n_classes = 101
        elif args.dataset == 'vox1':
            n_classes = 446
        elif args.dataset == 'vox2':
            n_classes = 5842
        else:
            raise NotImplementedError('Incorrect dataset name {}'.format(args.dataset))

        if fusion == 'sum':
            self.fusion_module = SumFusion(output_dim=n_classes)
        elif fusion == 'concat':
            self.fusion_module = ConcatFusion(output_dim=n_classes)
        elif fusion == 'film':
            self.fusion_module = FiLM(output_dim=n_classes, x_film=True)
        elif fusion == 'gated':
            self.fusion_module = GatedFusion(output_dim=n_classes, x_gate=True)
        else:
            raise NotImplementedError('Incorrect fusion method: {}!'.format(fusion))

        self.audio_net = resnet34_audio()
        self.visual_net = resnet34_visual()

        

    def forward(self, audio, visual,args):

        a = self.audio_net(audio)
        
        v = self.visual_net(visual)
        

        (_, C, H, W) = v.size()
        B = a.size()[0]
        v = v.view(B, -1, C, H, W)
        v = v.permute(0, 2, 1, 3, 4)

        v = F.adaptive_avg_pool3d(v, 1)

        a = torch.flatten(a, 1)
        v = torch.flatten(v, 1)

        a, v, out = self.fusion_module(a, v)

        return a, v, out




# class AVClassifier_LP(nn.Module):
#     def __init__(self, args):
#         super(AVClassifier_LP, self).__init__()

        
#         if args.dataset == 'VGGSound':
#             n_classes = 309
#         elif args.dataset == 'KineticSound':
#             n_classes = 31
#         elif args.dataset == 'CREMAD':
#             n_classes = 6
#         elif args.dataset == 'AVE':
#             n_classes = 28
#         elif args.dataset == 'vox1':
#             n_classes = 446
        
#         else:
#             raise NotImplementedError('Incorrect dataset name {}'.format(args.dataset))

        
        
#         self.audio_net = resnet18_audio()
#         self.visual_net = resnet18_visual()

#         self.LP_a=nn.Linear(512,n_classes)
#         self.LP_v=nn.Linear(512,n_classes)
        

#     def forward(self, audio, visual,args):
       
#         a=self.audio_net(audio)
#         v=self.visual_net(visual)
       
#         #print("size of a{}".format(a.size()))
#         #print("size of v{}".format(v.size()))
       

#         (_, C, H, W) = v.size()
#         B = a.size()[0]
#         v = v.view(B, -1, C, H, W)
#         v = v.permute(0, 2, 1, 3, 4)

#         #print("beforepool_a_v{}{}".format(a.size(),v.size()))


#         v = F.adaptive_avg_pool3d(v, 1)
#         a = F.adaptive_avg_pool2d(a, 1)
#         v=v.squeeze(2).squeeze(2).squeeze(2)
#         a=a.squeeze(2).squeeze(2)

#         #print("afterpool_a_v{}{}".format(a.size(),v.size()))

#         a=self.LP_a(a)
#         v=self.LP_v(v)
#         out_a=a
#         out_v=v

#         return out_a, out_v


