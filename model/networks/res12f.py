# from cvxpy import norm1
import torch.nn as nn
import torch
import torch.nn.functional as F
from model.networks.dropblock import DropBlock
import torchjpeg.dct as dctt
# This ResNet network was designed following the practice of the following papers:
# TADAM: Task dependent adaptive metric for improved few-shot learning (Oreshkin et al., in NIPS 2018) and
# A Simple Neural Attentive Meta-Learner (Mishra et al., in ICLR 2018).
import torch.fft
# import math

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

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

class Freq_Attention(nn.Module):
    def __init__(self, num=100):
        super().__init__()
        self.freq_att = nn.Parameter(torch.randn(num,64))

    def forward(self, x):

        # convert images to dct with same size
        x_shot_f = dctt.images_to_batch(x) #(b*N*K, C, H, W)

        # convert dct to zigzag
        x_shot_z = dctt.zigzag(x_shot_f) #(b*N*K, C, L, 64)

        # frequency attention
        self.freq_att = self.freq_att.to(x_shot_z.device)
        x_shot_za = x_shot_z*self.freq_att #(b*N*K, C, L, 64)  

        # x_tmp = x_shot_z.view(x_shot_z.shape[0], -1, x_shot_z.shape[-1]).permute(0, 2, 1)
        # self.freq_att = self.freq_net(x_tmp, x_tmp, x_tmp).permute(0, 1, 2)

        # x_shot_za = self.freq_att.view(*x_shot_z.shape)

        # convert zigzag to dct
        _, ind = torch.sort(zigzag_indices)
        x_shot_iza = x_shot_za[..., ind].view(x_shot_za.shape[0], x_shot_za.shape[1], x_shot_za.shape[2], 8, 8) #(b*N*K, C, L, 8, 8)

        # deblockify
        x_shot_iza = dctt.deblockify(x_shot_iza,(x.shape[-2], x.shape[-1])) #(b*N*K, C, H, W)

        # convert dct to images
        x_shot_out = dctt.batch_to_images(x_shot_iza) #(b*N*K, C, H, W)

        return x_shot_out

class FreqFilter(nn.Module):
    def __init__(self, dim=640, h=5, w=5):
        super().__init__()
        # self.complex_weight = nn.Parameter(torch.randn(dim, h, w, 2, dtype=torch.float32) * 0.02)

        self.freq_att = nn.Parameter(torch.randn(dim,h,w))

    def forward(self, x):
        # B, C, H, W = x.shape
        # # x = torch.fft.rfft2(x, dim=(2, 3), norm='ortho')

        # x = torch.rfft(x, signal_ndim=2, normalized=False, onesided=False)
        # # weight = torch.view_as_complex(self.complex_weight)
        # weight = self.complex_weight
        # x = x * weight
        # # x = torch.fft.irfft2(x, s=(H, W), dim=(2, 3), norm='ortho')
        # x = torch.irfft(x, signal_ndim=2, normalized=False, onesided=False)

        x_dct = dctt.batch_dct(x) # (b, C, H, W)
        x = dctt.batch_idct(x_dct * self.freq_att) # (b, C, H, W)

        return x


class GlobalFilter(nn.Module):
    def __init__(self, dim=640, h=5, w=5):
        super().__init__()
        self.complex_weight = nn.Parameter(torch.randn(h, w, dim, 2, dtype=torch.float32) * 0.02)

    def forward(self, x):
        B, H, W, C = x.shape
        x = torch.fft.rfft2(x, dim=(1, 2), norm='ortho')
        weight = torch.view_as_complex(self.complex_weight)
        x = x * weight
        x = torch.fft.irfft2(x, s=(H, W), dim=(1, 2), norm='ortho')
        return x

class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1) # F_squeeze 
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class SEBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, drop_rate=0.0, drop_block=False, block_size=1):
        super(SEBasicBlock, self).__init__()
        self.se = SELayer(planes, reduction=16)
        self.conv1 = conv3x3(inplanes, planes)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.LeakyReLU(0.1)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = conv3x3(planes, planes)
        self.bn3 = nn.BatchNorm2d(planes)
        self.maxpool = nn.MaxPool2d(stride)
        self.downsample = downsample
        self.stride = stride
        self.drop_rate = drop_rate
        self.num_batches_tracked = 0
        self.drop_block = drop_block
        self.block_size = block_size


    def forward(self, x):
        self.num_batches_tracked += 1

        residual = x

        # print(residual.shape)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        # out = tdct.dct_2d(out)
        # out = self.se(out)
        # out = tdct.idct_2d(out)

        out = dctt.images_to_batch(out)
        out = self.se(out)
        out = dctt.batch_to_images(out)

        # print(out.shape)

        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        out = self.maxpool(out)

        if self.drop_rate > 0:
            out = F.dropout(out, p=self.drop_rate, training=self.training, inplace=True)

        return out

class SEBasicBlock_o(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, drop_rate=0.0, drop_block=False, block_size=1):
        super(SEBasicBlock_o, self).__init__()
        self.se = SELayer(planes, reduction=16)
        self.conv1 = conv3x3(inplanes, planes)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.LeakyReLU(0.1)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = conv3x3(planes, planes)
        self.bn3 = nn.BatchNorm2d(planes)
        self.maxpool = nn.MaxPool2d(stride)
        self.downsample = downsample
        self.stride = stride
        self.drop_rate = drop_rate
        self.num_batches_tracked = 0
        self.drop_block = drop_block
        self.block_size = block_size


    def forward(self, x):
        self.num_batches_tracked += 1

        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        # out = tdct.dct_2d(out)
        # out = self.se(out)
        # out = tdct.idct_2d(out)

        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        out = self.maxpool(out)

        if self.drop_rate > 0:
            out = F.dropout(out, p=self.drop_rate, training=self.training, inplace=True)

        return out

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, drop_rate=0.0, drop_block=False, block_size=1):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.LeakyReLU(0.1)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = conv3x3(planes, planes)
        self.bn3 = nn.BatchNorm2d(planes)
        self.maxpool = nn.MaxPool2d(stride)
        self.downsample = downsample
        self.stride = stride
        self.drop_rate = drop_rate
        self.num_batches_tracked = 0
        self.drop_block = drop_block
        self.block_size = block_size
        self.DropBlock = DropBlock(block_size=self.block_size)

    def forward(self, x):
        self.num_batches_tracked += 1

        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        out = self.maxpool(out)

        if self.drop_rate > 0:
            if self.drop_block == True:
                feat_size = out.size()[2]
                keep_rate = max(1.0 - self.drop_rate / (20*2000) * (self.num_batches_tracked), 1.0 - self.drop_rate)
                gamma = (1 - keep_rate) / self.block_size**2 * feat_size**2 / (feat_size - self.block_size + 1)**2
                out = self.DropBlock(out, gamma=gamma)
            else:
                out = F.dropout(out, p=self.drop_rate, training=self.training, inplace=True)

        return out

class ResNet(nn.Module):
    
    def __init__(self, block=BasicBlock, keep_prob=1.0, avg_pool=True, drop_rate=0.1, dropblock_size=5):
        self.inplanes = 3
        super(ResNet, self).__init__()

        self.layer1 = self._make_layer(block, 64, stride=2, drop_rate=drop_rate)
        self.layer2 = self._make_layer(block, 160, stride=2, drop_rate=drop_rate)
        self.layer3 = self._make_layer(block, 320, stride=2, drop_rate=drop_rate, drop_block=True, block_size=dropblock_size)
        self.layer4 = self._make_layer(block, 640, stride=2, drop_rate=drop_rate, drop_block=True, block_size=dropblock_size)
        if avg_pool:
            self.avgpool = nn.AvgPool2d(5, stride=1)
        self.keep_prob = keep_prob
        self.keep_avg_pool = avg_pool
        self.dropout = nn.Dropout(p=1 - self.keep_prob, inplace=False)
        self.drop_rate = drop_rate

        # self.freqfilt1 = FreqFilter(3, 80, 80)
        # self.freqfilt1 = FreqFilter(64,42,42)
        # self.freqfilt2 = FreqFilter(160,21,21)
        # self.freqfilt1 = FreqFilter(64,40,40)
        # self.freqfilt2 = FreqFilter(160,20,20)

        # self.bnff1 = nn.BatchNorm2d(64)
        # self.bnff2 = nn.BatchNorm2d(160)

        # self.freqfilt3 = FreqFilter(320,10,10)
        # self.freqfilt4 = FreqFilter(640,5,5)

        # self.freqfilt1 = Freq_Attention(num=100)
        # self.freqfilt2 = Freq_Attention(num=25)
        # self.freqfilt3 = Freq_Attention(num=4)
        # self.freqfilt4 = Freq_Attention(num=1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, stride=1, drop_rate=0.0, drop_block=False, block_size=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, drop_rate, drop_block, block_size))
        self.inplanes = planes * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x):

        # x = self.freqfilt1(x)
        x = self.layer1(x) # b*64*42*42
        # x = self.freqfilt1(x)
        # x = self.bnff1(x)

        x = self.layer2(x) # b*160*21*21
        # x = self.freqfilt2(x)
        # x = self.bnff2(x)
        
        # x = self.freqfilt3(x)
        x = self.layer3(x) # b*320*10*10
        
        # x = self.freqfilt4(x)
        m = self.layer4(x) # b*640*5*5
        

        if self.keep_avg_pool:
            x = self.avgpool(m)
        x = x.view(x.size(0), -1)
        # print(m.shape)
        # print(x.shape)
        return m, x
        # return x

class ResNet_se(nn.Module):
    
    def __init__(self, args,block=SEBasicBlock, keep_prob=1.0, avg_pool=False, drop_rate=0.0, dropblock_size=5):
        self.inplanes = 3
        super(ResNet_se, self).__init__()

        self.layer1 = self._make_layer(block, 64, stride=2, drop_rate=drop_rate)
        self.layer2 = self._make_layer(block, 160, stride=2, drop_rate=drop_rate)
        self.layer3 = self._make_layer(block, 320, stride=2, drop_rate=drop_rate, drop_block=True,
                                       block_size=dropblock_size)
        self.layer4 = self._make_layer(block, 640, stride=2, drop_rate=drop_rate, drop_block=True,
                                       block_size=dropblock_size)
        if avg_pool:
            self.avgpool = nn.AvgPool2d(5, stride=1)
        self.keep_prob = keep_prob
        self.keep_avg_pool = avg_pool
        self.dropout = nn.Dropout(p=1 - self.keep_prob, inplace=False)
        self.drop_rate = drop_rate

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, stride=1, drop_rate=0.0, drop_block=False, block_size=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, drop_rate, drop_block, block_size))
        self.inplanes = planes * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.layer1(x)

        x = self.layer2(x)

        x = self.layer3(x)

        m = self.layer4(x)
        if self.keep_avg_pool:
            x = self.avgpool(m)
        x = x.view(x.size(0), -1)
        # print(m.shape)
        # print(x.shape)
        return m, x

        # return x

def Res12f(keep_prob=1.0, avg_pool=True, **kwargs):
    """Constructs a ResNet-12 model.
    """
    model = ResNet(BasicBlock, keep_prob=keep_prob, avg_pool=avg_pool, **kwargs)
    return model
