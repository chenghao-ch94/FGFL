import torch
import torch.nn as nn
import numpy as np

import torchjpeg.dct as dctt
from torchvision import transforms
import numpy as np
import torch_dct as dctt2

zigzag_indices = torch.tensor([
        0,  1,  5,  6, 14, 15, 27, 28,
        2,  4,  7, 13, 16, 26, 29, 42,
        3,  8, 12, 17, 25, 30, 41, 43,
        9, 11, 18, 24, 31, 40, 44, 53,
    10, 19, 23, 32, 39, 45, 52, 54,
    20, 22, 33, 38, 46, 51, 55, 60,
    21, 34, 37, 47, 50, 56, 59, 61,
    35, 36, 48, 49, 57, 58, 62, 63 
]).long() 


class FewShotModel(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        if args.backbone_class == 'ConvNet':
            from model.networks.convnet import ConvNet
            self.encoder = ConvNet()
        elif args.backbone_class == 'Res12':
            hdim = 640
            from model.networks.res12 import ResNet
            self.encoder = ResNet()
        else:
            raise ValueError('')

        self.mean1 = (120.39586422/255.0,  115.59361427/255.0, 104.54012653/255.0)
        self.std1 = (70.68188272/255.0,   68.27635443/255.0,  72.54505529/255.0)
               
        self.mean = (-1.5739, -0.8470,  0.4505)
        self.std = (66.6648, 20.2999, 18.2193)

        self.img_size = 84

    def freq_att_forward(self, x):
        # convert images to dct with same size
        x_shot_f = dctt.images_to_batch(x) #(b*N*K, C, H, W)

        # convert dct to zigzag
        x_shot_z = dctt.zigzag(x_shot_f) #(b*N*K, C, L, 64)

        # frequency attention
        self.freq_att = self.freq_att.to(x_shot_z.device)
        x_shot_za = x_shot_z*self.freq_att #(b*N*K, C, L, 64)  

        # convert zigzag to dct
        _, ind = torch.sort(zigzag_indices)
        x_shot_iza = x_shot_za[..., ind].view(x_shot_za.shape[0], x_shot_za.shape[1], x_shot_za.shape[2], 8, 8) #(b*N*K, C, L, 8, 8)

        # deblockify
        x_shot_iza = dctt.deblockify(x_shot_iza,(x.shape[-2], x.shape[-1])) #(b*N*K, C, H, W)

        # convert dct to images
        x_shot_out = dctt.batch_to_images(x_shot_iza) #(b*N*K, C, H, W)

        return x_shot_out

    def split_instances(self, data):
        args = self.args
        if self.training:
            return  (torch.Tensor(np.arange(args.way*args.shot)).long().view(1, args.shot, args.way), 
                     torch.Tensor(np.arange(args.way*args.shot, args.way * (args.shot + args.query))).long().view(1, args.query, args.way))
        else:
            return  (torch.Tensor(np.arange(args.eval_way*args.eval_shot)).long().view(1, args.eval_shot, args.eval_way), 
                     torch.Tensor(np.arange(args.eval_way*args.eval_shot, args.eval_way * (args.eval_shot + args.eval_query))).long().view(1, args.eval_query, args.eval_way))

    def denorm(self, tensor):
        t_mean = torch.FloatTensor(self.mean).view(3,1,1).expand(3,self.img_size,self.img_size).cuda(device=tensor.device)
        t_std = torch.FloatTensor(self.std).view(3,1,1).expand(3,self.img_size,self.img_size).cuda(device=tensor.device)
        return tensor * t_std + t_mean

    def fnorm(self, tensor):
        t_mean = torch.FloatTensor(self.mean).view(3,1,1).expand(3,self.img_size,self.img_size).cuda(device=tensor.device)
        t_std = torch.FloatTensor(self.std).view(3,1,1).expand(3,self.img_size,self.img_size).cuda(device=tensor.device)
        return (tensor - t_mean) / t_std

    def denorms(self, tensor):
        t_mean = torch.FloatTensor(self.mean1).view(3,1,1).expand(3,self.img_size,self.img_size).cuda(device=tensor.device)
        t_std = torch.FloatTensor(self.std1).view(3,1,1).expand(3,self.img_size,self.img_size).cuda(device=tensor.device)
        return tensor * t_std + t_mean

    def fnorms(self, tensor):
        t_mean = torch.FloatTensor(self.mean1).view(3,1,1).expand(3,self.img_size,self.img_size).cuda(device=tensor.device)
        t_std = torch.FloatTensor(self.std1).view(3,1,1).expand(3,self.img_size,self.img_size).cuda(device=tensor.device)
        return (tensor - t_mean) / t_std


    def forward(self, x, get_feature=False):
  
        if get_feature:

            x = dctt.images_to_batch(self.denorms(x))
            _, instance_embs = self.encoder(x)
            return instance_embs

        else:

            x = dctt.images_to_batch(self.denorms(x))
            _, instance_embs = self.encoder(x)
            
            support_idx, query_idx = self.split_instances(x)

        if self.training:
           
            logits, logits_reg = self._forward(instance_embs, support_idx, query_idx)
           
            return logits, logits_reg
        
        else:
          
            logits = self._forward(instance_embs, support_idx, query_idx)
           
            return logits
        
    def _forward(self, x, support_idx, query_idx):
        raise NotImplementedError('Suppose to be implemented by subclass')