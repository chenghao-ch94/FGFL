import torch
import torch.nn as nn
import numpy as np
from model.utils import euclidean_metric
import torch.nn.functional as F
from torchvision import transforms
import torch_dct as dctt2
import torchjpeg.dct as dctt

def is_bn(m):
    return isinstance(m, nn.modules.batchnorm.BatchNorm2d) | isinstance(m, nn.modules.batchnorm.BatchNorm1d)

def take_bn_layers(model):
    for m in model.modules():
        if is_bn(m):
            yield m
    return []

class FreezedBnModel(nn.Module):
    def __init__(self, model, is_train=True):
        super(FreezedBnModel, self).__init__()
        self.model = model
        self.bn_layers = list(take_bn_layers(self.model))


    def forward(self, x):
        is_train = len(self.bn_layers) > 0 and self.bn_layers[0].training
        if is_train:
            self.set_bn_train_status(is_train=False)
        predicted = self.model(x)
        if is_train:
            self.set_bn_train_status(is_train=True)

        return predicted

    def set_bn_train_status(self, is_train: bool):
        for layer in self.bn_layers:
            layer.train(mode=is_train)
            layer.weight.requires_grad = is_train #TODO: layer.requires_grad = is_train - check is its OK
            layer.bias.requires_grad = is_train

class Classifier(nn.Module):

    def __init__(self, args):
        super().__init__()
        self.args = args
        if args.backbone_class == 'ConvNet':
            from model.networks.convnet import ConvNet
            hdim = 64
            self.encoder = ConvNet()
        elif args.backbone_class == 'Res12':
            hdim = 640
            from model.networks.res12 import ResNet
            self.encoder = ResNet()
        elif args.backbone_class == 'Res18':
            hdim = 512
            from model.networks.res18 import ResNet
            self.encoder = ResNet()
        elif args.backbone_class == 'WRN':
            hdim = 640
            from model.networks.WRN28 import Wide_ResNet
            self.encoder = Wide_ResNet(28, 10, 0.5)  
        elif args.backbone_class == 'Res12_bdc':
            # hdim = 640
            hdim = int(640 * (640+1) / 2)
            from model.networks.res12_new import ResNet
            self.encoder = ResNet()   
        elif args.backbone_class == 'Res18_bdc':
            # hdim = 640
            hdim = int(256 * (256+1) / 2)
            from model.networks.res18_new import ResNet
            self.encoder = ResNet()             
        else:
            raise ValueError('')

        self.fc = nn.Linear(hdim, args.num_class)

        # self.mean1 = (0.485, 0.456, 0.406)
        # self.std1 = (0.229, 0.224, 0.225)

        self.mean1 =  (120.39586422 / 255.0,  115.59361427 / 255.0, 104.54012653/ 255.0)
        self.std1 = (70.68188272 / 255.0,   68.27635443 / 255.0,  72.54505529 / 255.0)
        
        # self.mean = (-1.5739, -0.8470,  0.4505)  #mini
        # self.std = (66.6648, 20.2999, 18.2193)

        self.mean = (-2.7477, 0.5098, -0.6573)    #tiered
        self.std = (71.5026, 16.1169, 15.5343)

        # self.grad_layer = 'encoder.layer4'

        # # Feed-forward features
        # self.feed_forward_features = None
        # # Backward features
        # self.backward_features = None
        # # Register hooks
        # self._register_hooks(self.grad_layer)

        self.img_size = 84

        # # sigma, omega for making the soft-mask
        # self.sigma = 0.5 
        # self.omega = 10

        # self.freezed_bn_model = FreezedBnModel(self.encoder)

    def forward(self, data, labels=None, is_emb = False):

        # data = self.fnorm(self.denorms(data)).detach()
        # data = self.denorms(data)
        # data = dctt.images_to_batch(self.fnorm(data))
        data = dctt.images_to_batch(data)
        # data = dctt2.dct_2d(data, norm='ortho').detach()
        # data = self.fnorm(data).detach()

        # x_shot_f = dctt.images_to_batch(x) #(b*N*K, C, H, W)
        # # convert dct to zigzag
        # x_shot_z = dctt.zigzag(x_shot_f) #(b*N*K, C, L, 64)

        _, out = self.encoder(data)
        if not is_emb:
            out = self.fc(out)
        return out
        # return self.gen_freq_mask(data, labels)


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

    def _register_hooks(self, grad_layer):
        def forward_hook(module, input, output):
            self.feed_forward_features = output
        
        def backward_hook(module, grad_input, grad_output):
            self.backward_features = grad_output[0]
        
        gradient_layer_found = False

        for idx, m in self.named_modules():
            if idx == grad_layer:
                m.register_forward_hook(forward_hook)
                m.register_backward_hook(backward_hook)
                print("Register forward hook !")
                print("Register backward hook !")
                gradient_layer_found = True
                break

        # for our own sanity, confirm its existence
        if not gradient_layer_found:
            raise AttributeError('Gradient layer %s not found in the internal model' % grad_layer)

    def _to_ohe(self, labels, classs=64):
        ohe = torch.zeros((labels.size(0), classs), requires_grad=True)
        for i, label in enumerate(labels):
            ohe[i, label] = 1
        ohe = torch.autograd.Variable(ohe)

        return ohe

    def gen_freq_mask(self, x, labels=None):
        
        _, instance_embsf = self.encoder(x)

        with torch.enable_grad():
            
            self.encoder.zero_grad()

            logits_fc = self.fc(instance_embsf)
            # logits_fsf = self.fproto_forward(instance_embsf, support_idx, query_idx, self.temp) #fs on freq domain
            q_ohe = self._to_ohe(labels, 64).cuda()
            gradient_q = (logits_fc * q_ohe).sum(dim=1) #dim=1

            # logits_fsf = self.fproto_forward(instance_embsf, support_idx, query_idx, self.temp) #fs on freq domain
            # q_ohe = self._to_ohe(labels, self.args.way).cuda()    
            # gradient_q = (logits_fsf * q_ohe).sum(dim=1) #dim=1

            gradient_q.backward(gradient=torch.ones_like(gradient_q), retain_graph=True)
            # gradient_q = logits_fsf * q_ohe
            # gradient_q.backward(gradient=gradient_q, retain_graph=True)
            self.encoder.zero_grad()

        backward_features = self.backward_features

        ### Gain on feature map (both spatial and frequency domain)
        fl = self.feed_forward_features
        weights = F.adaptive_avg_pool2d(backward_features, 1)
        Ac = torch.mul(fl, weights).sum(dim=1, keepdim=True)
        Ac = F.relu(Ac)
        Ac = F.interpolate(Ac, mode='bilinear', align_corners=True, size=(x.shape[-2], x.shape[-1]))

        # heatmap = Ac

        # Ac_min = Ac.min()
        # Ac_max = Ac.max()
        # # print(Ac_min, Ac_max)
        # scaled_ac = (Ac - Ac_min) / (Ac_max - Ac_min)
        # mask = torch.sigmoid(self.omega * (scaled_ac - self.sigma))
        # # mask = torch.sigmoid(self.omega * (Ac - self.sigma))
        # # mask = scaled_ac
        # # mask = Ac

        Ac_min, _ = Ac.view(len(x), -1).min(dim=1)
        Ac_max, _ = Ac.view(len(x), -1).max(dim=1)
        import sys
        eps = torch.tensor(sys.float_info.epsilon).to(x.device)
        scaled_ac = (Ac - Ac_min.view(-1, 1, 1, 1)) / \
                    (Ac_max.view(-1, 1, 1, 1) - Ac_min.view(-1, 1, 1, 1)
                     + eps.view(1, 1, 1, 1))
        mask = F.sigmoid(self.omega * (scaled_ac - self.sigma))
        masked_image = x - x * mask

        _, am_fea = self.freezed_bn_model(masked_image)
        logits_am = self.fc(am_fea)

        return logits_fc, logits_am

    def forward_proto(self, data_shot, data_query, way = None):
        if way is None:
            way = self.args.num_class

        # data_shot = dctt.images_to_batch(self.denorms(data_shot))
        # data_query = dctt.images_to_batch(self.denorms(data_query))

        # data_shot = dctt2.dct_2d(self.denorms(data_shot), norm='ortho').detach()
        # data_query = dctt2.dct_2d(self.denorms(data_query), norm='ortho').detach()

        # data_shot = self.fnorm(data_shot).detach()
        # data_query = self.fnorm(data_query).detach()

        # data_shot = dctt.images_to_batch(self.fnorm(data_shot))
        # data_query = dctt.images_to_batch(self.fnorm(data_query))

        data_shot = dctt.images_to_batch(data_shot)
        data_query = dctt.images_to_batch(data_query)

        _, proto = self.encoder(data_shot)
        proto = proto.reshape(self.args.shot, way, -1).mean(dim=0)
        _, query = self.encoder(data_query.contiguous())
        
        logits_dist = euclidean_metric(query, proto)
        logits_sim = torch.mm(query, F.normalize(proto, p=2, dim=-1).t())
        # logits_sim = torch.mm(F.normalize(query,p=2,dim=-1), F.normalize(proto, p=2, dim=-1).t())
        return logits_dist, logits_sim