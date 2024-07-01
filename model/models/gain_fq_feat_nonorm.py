import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchjpeg.dct as dctt
from model.utils import one_hot
import math

from torch.autograd import Function
class _ReverseGrad(Function):

    @staticmethod
    def forward(ctx, input, grad_scaling):
        ctx.grad_scaling = grad_scaling
        return input.view_as(input)

    @staticmethod
    def backward(ctx, grad_output):
        grad_scaling = ctx.grad_scaling
        return -grad_scaling * grad_output, None

reverse_grad = _ReverseGrad.apply

class ReverseGrad(nn.Module):
    """Gradient reversal layer.

    It acts as an identity layer in the forward,
    but reverses the sign of the gradient in
    the backward.
    """

    def forward(self, x, grad_scaling=1.):
        assert grad_scaling >= 0, \
            'grad_scaling must be non-negative, ' \
            'but got {}'.format(grad_scaling)
        return reverse_grad(x, grad_scaling)

class ScaledDotProductAttention(nn.Module):
    ''' Scaled Dot-Product Attention '''

    def __init__(self, temperature, attn_dropout=0.1, ba=0.5):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)
        self.softmax = nn.Softmax(dim=2)
        # self.ba = 0.5

        self.ba = nn.Parameter(torch.tensor(ba))

    def forward(self, q, k, v):

        attn = torch.bmm(q, k.transpose(1, 2))
        attn = attn / self.temperature
        attn = self.softmax(attn)
   
        attn = self.dropout(attn)
        output = torch.bmm(attn, v)

        return output

class MultiHeadAttention(nn.Module):
    ''' Multi-Head Attention module '''

    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1):
        super().__init__()
        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.w_qs = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_ks = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_vs = nn.Linear(d_model, n_head * d_v, bias=False)
        nn.init.normal_(self.w_qs.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_k)))
        nn.init.normal_(self.w_ks.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_k)))
        nn.init.normal_(self.w_vs.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_v)))

        self.attention = ScaledDotProductAttention(temperature=np.power(d_k, 0.5))
        self.layer_norm = nn.LayerNorm(d_model)

        self.fc = nn.Linear(n_head * d_v, d_model)
        nn.init.xavier_normal_(self.fc.weight)
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v):  # tag=True: q,v ->sp  k -> fq, tag=False: q,k,v -> sp
        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
        sz_b, len_q, _ = q.size()
        sz_b, len_k, _ = k.size()
        sz_b, len_v, _ = v.size()

        residual = q

        q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)
        k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
        v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)
    
        q = q.permute(2, 0, 1, 3).contiguous().view(-1, len_q, d_k) # (n*b) x lq x dk
        k = k.permute(2, 0, 1, 3).contiguous().view(-1, len_k, d_k) # (n*b) x lk x dk
        v = v.permute(2, 0, 1, 3).contiguous().view(-1, len_v, d_v) # (n*b) x lv x dv
        

        output = self.attention(q, k, v)

        output = output.view(n_head, sz_b, len_q, d_v)
        output = output.permute(1, 2, 0, 3).contiguous().view(sz_b, len_q, -1) # b x lq x (n*dv)

        output = self.dropout(self.fc(output))
        output = self.layer_norm(output + residual)

        return output

def count_acc(logits, label):
    pred = torch.argmax(logits, dim=1)
    if torch.cuda.is_available():
        return (pred == label).type(torch.cuda.FloatTensor).mean().item()
    else:
        return (pred == label).type(torch.FloatTensor).mean().item()

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

class GAINModel(nn.Module):
    def __init__(self, args):
        super(GAINModel, self).__init__()
        self.args = args
        if args.backbone_class == 'Res12':
            hdim = 640
            from model.networks.res12 import ResNet
            self.encoder = ResNet()
            self.encoder_f = ResNet()

        self.mean1 = (120.39586422/255.0,  115.59361427/255.0, 104.54012653/255.0)
        self.std1 = (70.68188272/255.0,   68.27635443/255.0,  72.54505529/255.0)
               
        self.mean = (-1.5739, -0.8470,  0.4505)
        self.std = (66.6648, 20.2999, 18.2193)

        self.grad_layer = 'encoder_f.layer4' # 'encoder_f.layer4.0.conv3' 

        # Feed-forward features
        self.feed_forward_features = None
        # Backward features
        self.backward_features = None
        # Register hooks
        self._register_hooks(self.grad_layer)

        self.img_size = 84

        self.reverse_layer = ReverseGrad()
        self.lambda_ = 0.5

        # sigma, omega for making the soft-mask
        self.sigma = 0.1  
        self.omega = 100
        self.temp =  12.5

        self.tri_loss_sp = nn.TripletMarginLoss(margin=0.1, p=2) #1-shot mini

        self.bn_layers_s = list(take_bn_layers(self.encoder))
        self.bn_layers_f = list(take_bn_layers(self.encoder_f))

        self.feat_dim = hdim
        self.slf_attn2 = MultiHeadAttention(1, hdim, hdim, hdim, dropout=0.5)

    def set_lambda(self, para):
        self.lambda_ = 2./(1+math.exp(-10.*para))-1
        return self.lambda_

    def freeze_forward(self, x, freq=False):
        
        if freq:
            bn_layers = self.bn_layers_f
        else:
            bn_layers = self.bn_layers_s

        is_train = len(bn_layers) > 0 and bn_layers[0].training
        if is_train:
            self.set_bn_train_status(listt = bn_layers, is_train=False)
    
        if freq:
            _, instance_embs = self.encoder_f(x)
        else:
           _, instance_embs = self.encoder(x)
    
        if is_train:
            self.set_bn_train_status(listt = bn_layers, is_train=True)

        return instance_embs

    def set_bn_train_status(self, listt, is_train: bool):
        for layer in listt:
            layer.train(mode=is_train)
            layer.weight.requires_grad = is_train #TODO: layer.requires_grad = is_train - check is its OK
            layer.bias.requires_grad = is_train

    def _register_hooks(self, grad_layer):
        def forward_hook(module, input, output):
            self.feed_forward_features = output

        def backward_hook(module, grad_input, grad_output):

            self.backward_features = grad_output[0]
        
        gradient_layer_found = False

        for idx, m in self.named_modules():
            # print(idx)
            if idx == grad_layer:
                m.register_forward_hook(forward_hook)
                # m.register_backward_hook(backward_hook)
                m.register_full_backward_hook(backward_hook) #avoid warning in the new pytorch version
                print("Register forward hook !")
                print("Register backward hook !")
                gradient_layer_found = True
                break

        # for our own sanity, confirm its existence
        if not gradient_layer_found:
            raise AttributeError('Gradient layer %s not found in the internal model' % grad_layer)

    def _to_ohe(self, labels, classs=5):

        ohe = torch.zeros((labels.size(0), classs))
        for i, label in enumerate(labels):
            ohe[i, label] = 1
        ohe = torch.autograd.Variable(ohe, requires_grad=True)

        return ohe

    def split_instances(self, data):
        args = self.args
        if self.training:
            return  (torch.Tensor(np.arange(args.way*args.shot)).long().view(1, args.shot, args.way), 
                     torch.Tensor(np.arange(args.way*args.shot, args.way * (args.shot + args.query))).long().view(1, args.query, args.way))
        else:
            return  (torch.Tensor(np.arange(args.eval_way*args.eval_shot)).long().view(1, args.eval_shot, args.eval_way), 
                     torch.Tensor(np.arange(args.eval_way*args.eval_shot, args.eval_way * (args.eval_shot + args.eval_query))).long().view(1, args.eval_query, args.eval_way))

    def prepare_label(self):
        args = self.args

        # prepare one-hot label
        if self.training:
            label = torch.arange(args.way, dtype=torch.int16).repeat(args.query)
            label_aux = torch.arange(args.way, dtype=torch.int8).repeat(args.shot + args.query)
            label_shot = torch.arange(args.way, dtype=torch.int16).repeat(args.shot)
        else:
            label = torch.arange(args.eval_way, dtype=torch.int16).repeat(args.eval_query)
            label_aux = torch.arange(args.eval_way, dtype=torch.int8).repeat(args.eval_shot + args.eval_query)
            label_shot = torch.arange(args.eval_way, dtype=torch.int16).repeat(args.eval_shot)
        
        label = label.type(torch.LongTensor)
        label_aux = label_aux.type(torch.LongTensor)
        label_shot = label_shot.type(torch.LongTensor)
        
        if torch.cuda.is_available():
            label = label.cuda()
            label_aux = label_aux.cuda()
            label_shot = label_shot.cuda()
            
        return label, label_aux, label_shot

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

    def forward(self, x, gt_lab=None, get_feature=False):

        return self.freq_forward(x, gt_lab)

    def loss_fn_kd(self, outputs, labels, teacher_outputs):
        """
        Compute the knowledge-distillation (KD) loss given outputs, labels.
        "Hyperparameters": temperature and alpha

        NOTE: the KL Divergence for PyTorch comparing the softmaxs of teacher
        and student expects the input tensor to be log probabilities! See Issue #2
        """

        alpha = 0.5
        T = 64

        KD_loss = nn.KLDivLoss()(F.log_softmax(outputs*self.temp/T, dim=1),
                                F.softmax(teacher_outputs*self.args.temperature2/T, dim=1)) * (alpha * T * T) + \
                F.cross_entropy(outputs, labels) * (1. - alpha)

        return KD_loss

    def freq_forward(self, x, gt_lab=None):

        q_lab, labels, s_lab = self.prepare_label()
        support_idx, query_idx = self.split_instances(x)
        _, instance_embs = self.encoder(x)
        x2 = dctt.images_to_batch(self.denorms(x).clamp(0,1)) #.detach()

        if self.training:
            # logits, logits_reg = self.feat_forward(instance_embs, support_idx, query_idx)
            logits, logits_reg = self.semi_protofeat(instance_embs, support_idx, query_idx)

            logits_fsf, freq_mask, logits_am, logits_am2 = self.gen_freq_mask(x2, s_lab, logits, q_lab)
        
            mask_x = self.fnorms(dctt.batch_to_images((x2*freq_mask).clamp(-128.0,127.0)).clamp(0,1)) #.detach()
            bad_x = self.fnorms(dctt.batch_to_images((x2*(1-freq_mask)).clamp(-128.0,127.0)).clamp(0,1)) #.detach()

            instance_embs_good = self.freeze_forward(mask_x, freq=False)
            instance_embs_bad = self.freeze_forward(bad_x, freq=False)

            loss_contr = self.tri_loss_sp(instance_embs, instance_embs_good, instance_embs_bad)                         #L_sw_tri

            instance_embs_bad = self.reverse_layer(instance_embs_bad.detach(), 0.1)

            logits_up = self.semi_protofeat_enh(instance_embs_good, instance_embs, support_idx, query_idx)                #L_aug
            # logits_up = self.semi_protofeat_enh_val(instance_embs_good, instance_embs, support_idx, query_idx)

            logits_sm, logits_qm = self.contrast_bad2(instance_embs_bad, instance_embs, support_idx, query_idx)          #L_cw_ctr

            # loss_distill = self.loss_fn_kd(logits_fsf, labels, logits_reg)                                                 #L_kd
            
            return logits, logits_reg, logits_fsf, logits_sm, logits_qm, loss_contr, logits_up, logits_am, logits_am2 #, loss_distill
            
        else:

            logits = self.semi_protofeat(instance_embs, support_idx, query_idx)

            return logits

    def gen_freq_mask(self, x, s_lab, probs, labels=None):
    
        _, instance_embsf = self.encoder_f(x)
        support_idx, query_idx = self.split_instances(x)

        with torch.enable_grad():
            
            self.encoder_f.zero_grad()

            logits_fsf = self.fproto_forward2(instance_embsf, support_idx, query_idx, self.temp) #fs on freq domain

            # print(logits_fsf.min(), logits_fsf.max())

            if self.training:
                q_ohe = self._to_ohe(s_lab, self.args.way).cuda()
                q_lab = self._to_ohe(labels, self.args.way).cuda()
                q_ohe = torch.cat([q_ohe, q_lab], dim=0)
                # q_ohe = torch.cat([q_ohe, probs.softmax(1)], dim=0)
                # q_ohe = probs.softmax(1)
            else:
                q_ohe = self._to_ohe(s_lab, self.args.eval_way).cuda()
                q_ohe = torch.cat([q_ohe, probs.softmax(1)], dim=0)

            # gradient = (logits_fsf / self.temp * q_ohe)
            gradient_q = (logits_fsf * self.temp * q_ohe).sum(dim=1)
            gradient_q.backward(gradient=torch.ones_like(gradient_q), retain_graph=True)
            # gradient_q.backward(gradient=gradient, retain_graph=True)

            self.encoder_f.zero_grad()

        backward_features = self.backward_features
        ### Gain on feature map (frequency domain)
        fl = self.feed_forward_features.to(x.device)
        weights = F.adaptive_avg_pool2d(backward_features, 1).to(x.device)

        Ac = torch.mul(fl, weights).sum(dim=1, keepdim=True)
        Ac = F.relu(Ac, inplace=True)
        Ac = F.interpolate(Ac, mode='bilinear', align_corners=True, size=(x.shape[-2], x.shape[-1]))

        Ac_min = Ac.min()
        Ac_max = Ac.max()
        scaled_ac = (Ac - Ac_min) / (Ac_max - Ac_min + 1e-8)
        mask = torch.sigmoid(self.omega * (scaled_ac - self.sigma)).to(x.device)

        if self.training:

            mask_embs = self.freeze_forward(x*(1-mask), freq=True)

            mask_embs = self.reverse_layer(mask_embs.detach(), self.lambda_)
            
            logits_am, logits_am2 = self.fproto_forward_pare(mask_embs, instance_embsf, support_idx, query_idx, self.temp) #fq: sup_ori -> sup_bad & query_bad

            return logits_fsf, mask, logits_am, logits_am2
        else:
            return logits_fsf, mask

    def forward_enhanced(self, instance_good, instance_embs, support_idx, query_idx):
    
        emb_dim = instance_embs.size(-1)
        # organize support/query data
        support = instance_embs[support_idx.contiguous().view(-1)].contiguous().view(*(support_idx.shape + (-1,)))
        query   = instance_embs[query_idx.contiguous().view(-1)].contiguous().view(  *(query_idx.shape   + (-1,)))

        support_good = instance_good[support_idx.contiguous().view(-1)].contiguous().view(*(support_idx.shape + (-1,)))
        query_good  = instance_good[query_idx.contiguous().view(-1)].contiguous().view(  *(query_idx.shape   + (-1,)))

        query = torch.cat([query, query_good], dim=0) # T x (K+Kq) x N x d
        support = torch.cat([support, support_good], dim=1) # T x (K+Kq) x N x d

        num_batch = support.shape[0]
        num_proto = self.args.way

        aux_task = torch.cat([support.view(1, self.args.shot*2, self.args.way, emb_dim), 
                            query.view(1, self.args.query*2, self.args.way, emb_dim)], 1) # T x (K+Kq) x N x d
        num_query = np.prod(aux_task.shape[1:3])
        aux_task = aux_task.permute([0, 2, 1, 3])
        aux_task = aux_task.contiguous().view(-1, self.args.shot*2 + self.args.query*2, emb_dim)

        aux_emb = self.slf_attn2(aux_task, aux_task, aux_task) # T x N x (K+Kq) x d
        # compute class mean
        aux_emb = aux_emb.view(num_batch, self.args.way, self.args.shot*2 + self.args.query*2, emb_dim)
        aux_center = torch.mean(aux_emb, 2) # T x N x d

        if self.training:
            if self.args.use_euclidean:
                aux_task = aux_task.permute([1,0,2]).contiguous().view(-1, emb_dim).unsqueeze(1) # (Nbatch*Nq*Nw, 1, d)
                aux_center = aux_center.unsqueeze(1).expand(num_batch, num_query, num_proto, emb_dim).contiguous()
                aux_center = aux_center.view(num_batch*num_query, num_proto, emb_dim) # (Nbatch x Nq, Nk, d)
                logits_reg = - torch.sum((aux_center - aux_task) ** 2, 2) / self.args.temperature2

            else:
                aux_center = F.normalize(aux_center, dim=-1) # normalize for cosine distance
                aux_task = aux_task.permute([1,0,2]).contiguous().view(num_batch, -1, emb_dim) # (Nbatch,  Nq*Nw, d)
                aux_task = F.normalize(aux_task, dim=-1) # normalize for cosine distance

                logits_reg = torch.bmm(aux_task, aux_center.permute([0,2,1])) / self.args.temperature2
                logits_reg = logits_reg.view(-1, num_proto)           
            
            return logits_reg
        
        else:
            if self.args.use_euclidean:
                aux_task = aux_task[:, self.args.shot*2:, :]
                aux_task = aux_task.permute([1,0,2]).contiguous().view(-1, emb_dim).unsqueeze(1) # (Nbatch*Nq*Nw, 1, d)
                aux_center = aux_center.unsqueeze(1).expand(num_batch, aux_task.shape[0], num_proto, emb_dim).contiguous()
                aux_center = aux_center.view(num_batch*aux_task.shape[0], num_proto, emb_dim) # (Nbatch x Nq, Nk, d)
                logits_reg = - torch.sum((aux_center - aux_task) ** 2, 2) / self.args.temperature2

            else:
                aux_center = F.normalize(aux_center, dim=-1) # normalize for cosine distance
                aux_task = aux_task[:, self.args.shot*2:, :]
                aux_task = aux_task.permute([1,0,2]).contiguous().view(num_batch, -1, emb_dim) # (Nbatch,  Nq*Nw, d)
                aux_task = F.normalize(aux_task, dim=-1) # normalize for cosine distance

                logits_reg = torch.bmm(aux_task, aux_center.permute([0,2,1])) / self.args.temperature2
                logits_reg = logits_reg.view(-1, num_proto)           
            
            return logits_reg            

    def contrast_bad2(self, instance_embs, instance_embs_ori, support_idx, query_idx):
        emb_dim = instance_embs.size(-1)

        # organize support/query data
        support_mask = instance_embs[support_idx.contiguous().view(-1)].contiguous().view(*(support_idx.shape + (-1,)))
        query_mask   = instance_embs[query_idx.contiguous().view(-1)].contiguous().view(  *(query_idx.shape   + (-1,)))

        support = instance_embs_ori[support_idx.contiguous().view(-1)].contiguous().view(*(support_idx.shape + (-1,)))
        query   = instance_embs_ori[query_idx.contiguous().view(-1)].contiguous().view(  *(query_idx.shape   + (-1,)))

        # get mean of the support
        # proto_mask = support_mask.mean(dim=1) # Ntask x NK x d
        # proto = support.mean(dim=1) # Ntask x NK x d

        num_batch = support.shape[0]
        num_query = np.prod(query_idx.shape[-2:])
        num_shot, num_way = support.shape[1], support.shape[2]

        whole_set = torch.cat([support.view(num_batch, -1, emb_dim), query.view(num_batch, -1, emb_dim)], 1)
        # support = self.slf_attn2(support.view(num_batch, -1, emb_dim), whole_set, whole_set).view(num_batch, num_shot, num_way, emb_dim)

        whole_set = self.slf_attn2(whole_set, whole_set, whole_set)
        support, query = whole_set.split([num_shot*num_way, num_query], 1)
        support = support.view(num_batch, num_shot, num_way, emb_dim)
        query = query.view(num_batch, -1, num_way, emb_dim)

        # get mean of the support
        proto = self.get_proto(support, query) # we can also use adapted query set here to achieve better results
        # proto = support.mean(dim=1) # Ntask x NK x d
        num_proto = proto.shape[1]

        whole_set_m = torch.cat([support_mask.view(num_batch, -1, emb_dim), query_mask.view(num_batch, -1, emb_dim)], 1)
        # support = self.slf_attn2(support.view(num_batch, -1, emb_dim), whole_set, whole_set).view(num_batch, num_shot, num_way, emb_dim)

        whole_set_m = self.slf_attn2(whole_set_m, whole_set_m, whole_set_m)
        support_mask, query_mask = whole_set_m.split([num_shot*num_way, num_query], 1)
        support_mask = support_mask.view(num_batch, num_shot, num_way, emb_dim)
        query_mask = query_mask.view(num_batch, -1, num_way, emb_dim)

        # get mean of the support
        proto_mask = self.get_proto(support_mask, query_mask) # we can also use adapted query set here to achieve better results
        # proto_mask = proto_mask.mean(dim=1) # Ntask x NK x d

        if self.args.use_euclidean:
            query = query.view(-1, emb_dim).unsqueeze(1) # (Nbatch*Nq*Nw, 1, d)
            query_mask = query_mask.view(-1, emb_dim).unsqueeze(1) # (Nbatch*Nq*Nw, 1, d)

            logits = - torch.sum((proto_mask - query) ** 2, 2) / self.args.temperature
            logits2 = - torch.sum((proto - query_mask) ** 2, 2) / self.args.temperature

        else: # cosine similarity: more memory efficient
            proto = F.normalize(proto, dim=-1) # normalize for cosine distance
            proto_mask = F.normalize(proto_mask, dim=-1) # normalize for cosine distance
            
            logits = torch.bmm(query, proto_mask.permute([0,2,1])) / self.args.temperature
            logits = logits.view(-1, num_proto)

            logits2 = torch.bmm(query_mask, proto.permute([0,2,1])) / self.args.temperature
            logits2 = logits2.view(-1, num_proto)
        
        return logits, logits2     

    def featstar(self, instance_embs, support_idx, query_idx):
        emb_dim = instance_embs.size(-1)

        # organize support/query data
        support = instance_embs[support_idx.contiguous().view(-1)].contiguous().view(*(support_idx.shape + (-1,)))
        query   = instance_embs[query_idx.contiguous().view(-1)].contiguous().view(  *(query_idx.shape   + (-1,)))
    
        # get mean of the support
        proto = support.mean(dim=1) # Ntask x NK x d
        num_batch = proto.shape[0]
        num_proto = proto.shape[1]
        num_query = np.prod(query_idx.shape[-2:])
    
        # query: (num_batch, num_query, num_proto, num_emb)
        # proto: (num_batch, num_proto, num_emb)
        query = query.view(-1, emb_dim).unsqueeze(1)

        proto = proto.unsqueeze(1).expand(num_batch, num_query, num_proto, emb_dim).contiguous()
        proto = proto.view(num_batch*num_query, num_proto, emb_dim)

        # refine by Transformer
        combined = torch.cat([proto, query], 1) # Nk x (N + 1) x d, batch_size = NK
        combined = self.slf_attn2(combined, combined, combined)
        # compute distance for all batches
        proto, query = combined.split(num_proto, 1)
        
        if self.args.use_euclidean:
            query = query.view(-1, emb_dim).unsqueeze(1) # (Nbatch*Nq*Nw, 1, d)

            logits = - torch.sum((proto - query) ** 2, 2) / self.args.temperature
        else: # cosine similarity: more memory efficient
            proto = F.normalize(proto, dim=-1) # normalize for cosine distance
            
            logits = torch.bmm(query, proto.permute([0,2,1])) / self.args.temperature
            logits = logits.view(-1, num_proto)
        
        if self.training:
            return logits, None
        else:
            return logits

    def get_proto(self, x_shot, x_pool):
        # get the prototypes based w/ an unlabeled pool set
        num_batch, num_shot, num_way, emb_dim = x_shot.shape
        num_pool_shot = x_pool.shape[1]
        num_pool = num_pool_shot * num_way
        label_support = torch.arange(num_way).repeat(num_shot).type(torch.LongTensor)
        label_support_onehot = one_hot(label_support, num_way)   
        label_support_onehot = label_support_onehot.unsqueeze(0).repeat([num_batch, 1, 1])
        if torch.cuda.is_available():
            label_support_onehot = label_support_onehot.cuda()
            
        proto_shot = x_shot.mean(dim = 1)
        if self.args.use_euclidean:
            dis = - torch.sum((proto_shot.unsqueeze(1).expand(num_batch, num_pool, num_way, emb_dim).contiguous().view(num_batch*num_pool, num_way, emb_dim) - x_pool.view(-1, emb_dim).unsqueeze(1)) ** 2, 2) / self.args.temperature
        else:
            dis = torch.bmm(x_pool.view(num_batch, -1, emb_dim), F.normalize(proto_shot, dim=-1).permute([0,2,1])) / self.args.temperature
                
        dis = dis.view(num_batch, -1, num_way)
        z_hat = F.softmax(dis, dim=2)
        z = torch.cat([label_support_onehot, z_hat], dim = 1)              # (num_batch, n_shot + n_pool, n_way)
        h = torch.cat([x_shot.view(num_batch, -1, emb_dim), x_pool.view(num_batch, -1, emb_dim)], dim = 1)        # (num_batch, n_shot + n_pool, n_embedding)

        proto = torch.bmm(z.permute([0,2,1]), h)
        sum_z = z.sum(dim = 1).view((num_batch, -1, 1))
        proto = proto / sum_z
        return proto        
        
    def semi_protofeat(self, instance_embs, support_idx, query_idx):
        emb_dim = instance_embs.size(-1)

        # organize support/query data
        support = instance_embs[support_idx.contiguous().view(-1)].contiguous().view(*(support_idx.shape + (-1,)))
        query   = instance_embs[query_idx.contiguous().view(-1)].contiguous().view(  *(query_idx.shape   + (-1,)))
    
        num_batch = support.shape[0]
        num_shot, num_way = support.shape[1], support.shape[2]
        num_query = np.prod(query_idx.shape[-2:])    
        
        # transformation
        whole_set = torch.cat([support.view(num_batch, -1, emb_dim), query.view(num_batch, -1, emb_dim)], 1)
        # support = self.slf_attn2(support.view(num_batch, -1, emb_dim), whole_set, whole_set).view(num_batch, num_shot, num_way, emb_dim)

        whole_set = self.slf_attn2(whole_set, whole_set, whole_set)
        support, query = whole_set.split([num_shot*num_way, num_query], 1)
        support = support.view(num_batch, num_shot, num_way, emb_dim)
        query = query.view(num_batch, -1, num_way, emb_dim)

        # get mean of the support
        proto = self.get_proto(support, query) # we can also use adapted query set here to achieve better results
        # proto = support.mean(dim=1) # Ntask x NK x d
        num_proto = proto.shape[1]
    
        # query: (num_batch, num_query, num_proto, num_emb)
        # proto: (num_batch, num_proto, num_emb)
        if self.args.use_euclidean:
            query = query.view(-1, emb_dim).unsqueeze(1) # (Nbatch*Nq*Nw, 1, d)
            proto = proto.unsqueeze(1).expand(num_batch, num_query, num_proto, emb_dim).contiguous()
            proto = proto.view(num_batch*num_query, num_proto, emb_dim) # (Nbatch x Nq, Nk, d)

            logits = - torch.sum((proto - query) ** 2, 2) / self.args.temperature
        else:
            proto = F.normalize(proto, dim=-1) # normalize for cosine distance
            query = query.view(num_batch, -1, emb_dim) # (Nbatch,  Nq*Nw, d)

            logits = torch.bmm(query, proto.permute([0,2,1])) / self.args.temperature
            logits = logits.view(-1, num_proto)
        
        # for regularization
        if self.training:
            aux_task = torch.cat([support.view(1, self.args.shot, self.args.way, emb_dim), 
                                  query.view(1, self.args.query, self.args.way, emb_dim)], 1) # T x (K+Kq) x N x d
            num_query = np.prod(aux_task.shape[1:3])
            aux_task = aux_task.permute([0, 2, 1, 3])
            aux_task = aux_task.contiguous().view(-1, self.args.shot + self.args.query, emb_dim)
            # apply the transformation over the Aug Task
            aux_emb = self.slf_attn2(aux_task, aux_task, aux_task) # T x N x (K+Kq) x d
            # compute class mean
            aux_emb = aux_emb.view(num_batch, self.args.way, self.args.shot + self.args.query, emb_dim)
            aux_center = torch.mean(aux_emb, 2) # T x N x d
            
            if self.args.use_euclidean:
                aux_task = aux_task.permute([1,0,2]).contiguous().view(-1, emb_dim).unsqueeze(1) # (Nbatch*Nq*Nw, 1, d)
                aux_center = aux_center.unsqueeze(1).expand(num_batch, num_query, num_proto, emb_dim).contiguous()
                aux_center = aux_center.view(num_batch*num_query, num_proto, emb_dim) # (Nbatch x Nq, Nk, d)
    
                logits_reg = - torch.sum((aux_center - aux_task) ** 2, 2) / self.args.temperature2
            else:
                aux_center = F.normalize(aux_center, dim=-1) # normalize for cosine distance
                aux_task = aux_task.permute([1,0,2]).contiguous().view(num_batch, -1, emb_dim) # (Nbatch,  Nq*Nw, d)
    
                logits_reg = torch.bmm(aux_task, aux_center.permute([0,2,1])) / self.args.temperature2
                logits_reg = logits_reg.view(-1, num_proto)            
            
            return logits, logits_reg      
            # return logits, None      
        else:
            return logits

    def semi_protofeat_enh(self, instance_good, instance_embs, support_idx, query_idx):
        
        emb_dim = instance_embs.size(-1)
        # organize support/query data
        support = instance_embs[support_idx.contiguous().view(-1)].contiguous().view(*(support_idx.shape + (-1,)))
        query   = instance_embs[query_idx.contiguous().view(-1)].contiguous().view(  *(query_idx.shape   + (-1,)))

        support_good = instance_good[support_idx.contiguous().view(-1)].contiguous().view(*(support_idx.shape + (-1,)))
        query_good = instance_good[query_idx.contiguous().view(-1)].contiguous().view(  *(query_idx.shape   + (-1,)))

        # support_new = torch.cat((support, support_good), dim=1)
        # proto = support_new.mean(dim=1) # Ntask x NK x d

        query = torch.cat([query, query_good], dim=1)
        # print(query.shape) # num_batch, 15*2, 5, emb_dim
        sup_enh = torch.cat([support, support_good], dim=1)
        # print(sup_enh.shape) # num_batch, 5*2, 5, emb_dim
    
        num_batch = support.shape[0]
        num_shot, num_way = sup_enh.shape[1], sup_enh.shape[2]
        num_query = np.prod(query_idx.shape[-2:])*2

        # whole_set = torch.cat([sup_enh.view(num_batch, -1, emb_dim), query.view(num_batch, -1, emb_dim)], 1)
        # support = self.slf_attn2(sup_enh.view(num_batch, -1, emb_dim), whole_set, whole_set).view(num_batch, num_shot, num_way, emb_dim)

        whole_set = torch.cat([sup_enh.view(num_batch, -1, emb_dim), query.view(num_batch, -1, emb_dim)], 1)
        # if self.training:
        whole_set = self.slf_attn2(whole_set, whole_set, whole_set) #.view(num_batch, num_shot, num_way, emb_dim)
        # else: # to-do: test
        #     whole_set = self.slf_attn2(whole_set, sup_enh.view(num_batch, -1, emb_dim), sup_enh.view(num_batch, -1, emb_dim))
        
        # print(whole_set.shape) # num_batch, 200, emb_dim

        support, query = whole_set.split([num_shot*num_way, num_query], 1)
        support = support.view(num_batch, num_shot, num_way, emb_dim)

        support, support_good = support.split([num_shot//2, num_shot//2], 1)
   
        query = query.view(num_batch, -1, num_way, emb_dim)
  
        query_enh = torch.cat([support_good, query], 1)
        
        # get mean of the support
        proto = self.get_proto(support, query_enh) # we can also use adapted query set here to achieve better results
        num_proto = proto.shape[1]

        if self.args.use_euclidean:
            query = query.view(-1, emb_dim).unsqueeze(1) # (Nbatch*Nq*Nw, 1, d)
            proto = proto.unsqueeze(1).expand(num_batch, num_query, num_proto, emb_dim).contiguous()
            proto = proto.view(num_batch*num_query, num_proto, emb_dim) # (Nbatch x Nq, Nk, d)

            logits = - torch.sum((proto - query) ** 2, 2) / self.args.temperature
        else:
            proto = F.normalize(proto, dim=-1) # normalize for cosine distance
            query = query.view(num_batch, -1, emb_dim) # (Nbatch,  Nq*Nw, d)

            logits = torch.bmm(query, proto.permute([0,2,1])) / self.args.temperature
            logits = logits.view(-1, num_proto)

        return logits

    def semi_protofeat_enh_val(self, instance_good, instance_embs, support_idx, query_idx):
        
        emb_dim = instance_embs.size(-1)
        # organize support/query data
        support = instance_embs[support_idx.contiguous().view(-1)].contiguous().view(*(support_idx.shape + (-1,)))
        query   = instance_embs[query_idx.contiguous().view(-1)].contiguous().view(  *(query_idx.shape   + (-1,)))

        support_good = instance_good[support_idx.contiguous().view(-1)].contiguous().view(*(support_idx.shape + (-1,)))
        query_good = instance_good[query_idx.contiguous().view(-1)].contiguous().view(  *(query_idx.shape   + (-1,)))

        # support_new = torch.cat((support, support_good), dim=1)
        # proto = support_new.mean(dim=1) # Ntask x NK x d

        # query = torch.cat([query, query_good], dim=1)

        if self.training:
            query = torch.cat([query, query_good], dim=0).mean(dim=0)

        # print(query.shape) # num_batch, 15*2, 5, emb_dim
        sup_enh = torch.cat([support, support_good], dim=1)
        # print(sup_enh.shape) # num_batch, 5*2, 5, emb_dim

        # sup_enh = support_good
    
        num_batch = support.shape[0]
        num_shot, num_way = sup_enh.shape[1], sup_enh.shape[2]
        num_query = np.prod(query_idx.shape[-2:]) #*2

        ##############################

        whole_set = torch.cat([sup_enh.view(num_batch, -1, emb_dim), query.view(num_batch, -1, emb_dim)], 1)
        whole_set = self.slf_attn2(whole_set, whole_set, whole_set)

        support, query = whole_set.split([num_shot*num_way, num_query], 1)
        support_enh = support.view(num_batch, num_shot, num_way, emb_dim)
   
        support, support_aug = support_enh.split([num_shot//2, num_shot//2], 1)
        support = support.view(num_batch, num_shot//2, num_way, emb_dim)
        support_aug = support_aug.view(num_batch, num_shot//2, num_way, emb_dim)

        # # whole_set = whole_set.view(num_batch, num_shot, num_way, emb_dim)

        proto = self.get_proto(support, support_aug)

        # proto = torch.mean(sup_enh, dim=1)
        # proto = self.slf_attn2(proto, proto, proto)

        #################################################

        num_proto = proto.shape[1]

        if self.args.use_euclidean:
            query = query.view(-1, emb_dim).unsqueeze(1) # (Nbatch*Nq*Nw, 1, d)
            proto = proto.unsqueeze(1).expand(num_batch, num_query, num_proto, emb_dim).contiguous()
            proto = proto.view(num_batch*num_query, num_proto, emb_dim) # (Nbatch x Nq, Nk, d)
            logits = - torch.sum((proto - query) ** 2, 2) / self.args.temperature
        else:
            proto = F.normalize(proto, dim=-1) # normalize for cosine distance
            query = query.view(num_batch, -1, emb_dim) # (Nbatch,  Nq*Nw, d)

            logits = torch.bmm(query, proto.permute([0,2,1])) / self.args.temperature
            logits = logits.view(-1, num_proto)

        return logits

    def semi_protofeat_enh_val2(self, instance_good, instance_embs, support_idx, query_idx):
        
        emb_dim = instance_embs.size(-1)
        # organize support/query data
        support = instance_embs[support_idx.contiguous().view(-1)].contiguous().view(*(support_idx.shape + (-1,)))
        query   = instance_embs[query_idx.contiguous().view(-1)].contiguous().view(  *(query_idx.shape   + (-1,)))

        support_good = instance_good[support_idx.contiguous().view(-1)].contiguous().view(*(support_idx.shape + (-1,)))

        proto = support.mean(dim=1) # Ntask x NK x d

        # query = torch.cat([query, query_good], dim=0)
        # print(query.shape) # num_batch*2, 15, 5, emb_dim
        sup_enh = torch.cat([support, support_good], dim=1)
        # print(sup_enh.shape) # num_batch, 5*2, 5, emb_dim

        num_shot, num_way = support.shape[1], support.shape[2]

        query = query.view(query.shape[0], -1, emb_dim) # num_batch*2, 15*5, emb_dim

        sup_enh = sup_enh.expand(query.shape[1], -1, -1, -1) # 15*5, 5 or 1 *2, 5, emb_dim

        whole_set = torch.cat([sup_enh.view(sup_enh.shape[0], -1, emb_dim), query.transpose(0,1)], 1) # 15*5, num_batch*(5*(1 or 5)*2+1), emb_dim

        whole_set = self.slf_attn2(whole_set, whole_set, whole_set)

        support, support_good, query = whole_set.split([num_shot*num_way, num_shot*num_way, query.shape[0]], 1) # 15*5, num_batch*(5*2), emb_dim ;  15*5, 2, emb_dim

        support = support.contiguous().view(-1, num_shot, num_way, emb_dim)

        support_good = support_good.contiguous().view(-1, num_shot, num_way, emb_dim)
        proto = self.get_proto(support, support_good)

        query = query.view(-1, emb_dim).unsqueeze(1) # (Nbatch*Nq*Nw, 1, d)
        logits = - torch.sum((proto - query) ** 2, 2) / self.args.temperature

        return logits

    def feat_forward(self, instance_embs, support_idx, query_idx):
        emb_dim = instance_embs.size(-1)

        # organize support/query data
        support = instance_embs[support_idx.contiguous().view(-1)].contiguous().view(*(support_idx.shape + (-1,)))
        query   = instance_embs[query_idx.contiguous().view(-1)].contiguous().view(  *(query_idx.shape   + (-1,)))

        num_batch = support.shape[0]
        num_shot, num_way = support.shape[1], support.shape[2]
        num_query = np.prod(query_idx.shape[-2:])
    
        proto = support.mean(dim=1) # Ntask x NK x d
        proto = self.slf_attn2(proto, proto, proto)
        num_proto = proto.shape[1]

        if self.args.use_euclidean:
            query = query.view(-1, emb_dim).unsqueeze(1) # (Nbatch*Nq*Nw, 1, d)
            proto = proto.unsqueeze(1).expand(num_batch, num_query, num_proto, emb_dim).contiguous()
            proto = proto.view(num_batch*num_query, num_proto, emb_dim) # (Nbatch x Nq, Nk, d)

            logits = - torch.sum((proto - query) ** 2, 2) / self.args.temperature
        else:
            proto = F.normalize(proto, dim=-1) # normalize for cosine distance
            query = query.view(num_batch, -1, emb_dim) # (Nbatch,  Nq*Nw, d)

            logits = torch.bmm(query, proto.permute([0,2,1])) / self.args.temperature
            logits = logits.view(-1, num_proto)
        
        # for regularization
        if self.training:
            aux_task = torch.cat([support.view(1, self.args.shot, self.args.way, emb_dim), 
                                  query.view(1, self.args.query, self.args.way, emb_dim)], 1) # T x (K+Kq) x N x d
            num_query = np.prod(aux_task.shape[1:3])
            aux_task = aux_task.permute([0, 2, 1, 3])
            aux_task = aux_task.contiguous().view(-1, self.args.shot + self.args.query, emb_dim)
            # apply the transformation over the Aug Task
            aux_emb = self.slf_attn2(aux_task, aux_task, aux_task) # T x N x (K+Kq) x d
            # compute class mean
            aux_emb = aux_emb.view(num_batch, self.args.way, self.args.shot + self.args.query, emb_dim)
            aux_center = torch.mean(aux_emb, 2) # T x N x d
            
            if self.args.use_euclidean:
                aux_task = aux_task.permute([1,0,2]).contiguous().view(-1, emb_dim).unsqueeze(1) # (Nbatch*Nq*Nw, 1, d)
                aux_center = aux_center.unsqueeze(1).expand(num_batch, num_query, num_proto, emb_dim).contiguous()
                aux_center = aux_center.view(num_batch*num_query, num_proto, emb_dim) # (Nbatch x Nq, Nk, d)
    
                logits_reg = - torch.sum((aux_center - aux_task) ** 2, 2) / self.args.temperature2
            else:
                aux_center = F.normalize(aux_center, dim=-1) # normalize for cosine distance
                aux_task = aux_task.permute([1,0,2]).contiguous().view(num_batch, -1, emb_dim) # (Nbatch,  Nq*Nw, d)
    
                logits_reg = torch.bmm(aux_task, aux_center.permute([0,2,1])) / self.args.temperature2
                logits_reg = logits_reg.view(-1, num_proto)        

            return logits, logits_reg       
                     
        else:
            return logits 

    def fproto_forward2(self, instance_embs, support_idx, query_idx, temp=64.):
        emb_dim = instance_embs.size(-1)

        # organize support/query data
        support = instance_embs[support_idx.flatten()].view(*(support_idx.shape + (-1,)))
        query   = instance_embs[query_idx.flatten()].view(  *(query_idx.shape   + (-1,)))

        # get mean of the support
        proto = support.mean(dim=1) #.detach() # Ntask x NK x d
        num_batch = proto.shape[0]
        num_proto = proto.shape[1]

        aux_task = torch.cat([support.view(1, self.args.shot, self.args.way, emb_dim), 
                                query.view(1, self.args.query, self.args.way, emb_dim)], 1) # T x (K+Kq) x N x d
        aux_task = aux_task.permute([0, 2, 1, 3])
        aux_task = aux_task.contiguous().view(-1, self.args.shot + self.args.query, emb_dim)
        aux_task = aux_task.permute([1,0,2]).contiguous().view(num_batch, -1, emb_dim)

        proto = F.normalize(proto, dim=-1)
        # aux_task = aux_task.permute([1,0,2]).contiguous().view(num_batch, -1, emb_dim)
        # aux_task = instance_embs.view(num_batch, -1, emb_dim)
        # aux_task = F.normalize(aux_task, dim=-1)
        logits = torch.bmm(aux_task, proto.permute([0,2,1])) / temp #/ self.args.temperature
        logits = logits.view(-1, num_proto)  

        return logits

    def fproto_forward_pare(self, instance_embs_bad, instance_embs, support_idx, query_idx, temp=64.):
        emb_dim = instance_embs.size(-1)

        # organize support/query data
        support = instance_embs[support_idx.flatten()].view(*(support_idx.shape + (-1,)))
        support_bad = instance_embs_bad[support_idx.flatten()].view(*(support_idx.shape + (-1,)))

        # # get mean of the support
        # proto = support.mean(dim=1) #.detach() # Ntask x NK x d
        # num_batch = proto.shape[0]
        # num_proto = proto.shape[1]

        # # proto = self.slf_attnf(proto, proto, proto)

        # proto = F.normalize(proto, dim=-1) # normalize for cosine distance
        # query = instance_embs_bad.unsqueeze(0)

        # # (num_batch,  num_emb, num_proto) * (num_batch, num_query*num_proto, num_emb) -> (num_batch, num_query*num_proto, num_proto)
        # logits = torch.bmm(query, proto.permute([0,2,1])) / self.args.temperature
        # logits = logits.view(-1, num_proto)

        # get mean of the support
        proto_bad = support_bad.mean(dim=1) #.detach() # Ntask x NK x d
        proto = support.mean(dim=1) #.detach() # Ntask x NK x d
        num_batch = proto.shape[0]
        num_proto = proto.shape[1]

        query_bad = instance_embs_bad[query_idx.flatten()].view(  *(query_idx.shape   + (-1,)))
        query   = instance_embs[query_idx.flatten()].view(  *(query_idx.shape   + (-1,)))

        aux_task_g = torch.cat([support.view(1, self.args.shot, self.args.way, emb_dim), 
                                query.view(1, self.args.query, self.args.way, emb_dim)], 1) # T x (K+Kq) x N x d
        aux_task_g = aux_task_g.permute([0, 2, 1, 3])
        aux_task_g = aux_task_g.contiguous().view(-1, self.args.shot + self.args.query, emb_dim)
        aux_task_g = aux_task_g.permute([1,0,2]).contiguous().view(num_batch, -1, emb_dim)

        aux_task_b = torch.cat([support_bad.view(1, self.args.shot, self.args.way, emb_dim), 
                                query_bad.view(1, self.args.query, self.args.way, emb_dim)], 1) # T x (K+Kq) x N x d
        aux_task_b = aux_task_b.permute([0, 2, 1, 3])
        aux_task_b = aux_task_b.contiguous().view(-1, self.args.shot + self.args.query, emb_dim)
        aux_task_b = aux_task_b.permute([1,0,2]).contiguous().view(num_batch, -1, emb_dim)


        proto_bad = F.normalize(proto_bad, dim=-1) # normalize for cosine distance
        # query = instance_embs.unsqueeze(0)
        # (num_batch,  num_emb, num_proto) * (num_batch, num_query*num_proto, num_emb) -> (num_batch, num_query*num_proto, num_proto)
        logits = torch.bmm(aux_task_g, proto_bad.permute([0,2,1])) / temp #self.args.temperature
        logits = logits.view(-1, num_proto)

        proto = F.normalize(proto, dim=-1) # normalize for cosine distance
        # query_bad = instance_embs_bad.unsqueeze(0)
        logits2 = torch.bmm(aux_task_b, proto.permute([0,2,1])) / temp #self.args.temperature
        logits2 = logits2.view(-1, num_proto)
     
        return logits, logits2
