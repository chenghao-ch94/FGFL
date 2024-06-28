from __future__ import print_function

import os
import os.path as osp
import numpy as np
import pickle
import sys
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
from PIL import Image

import torch_dct as dct
import torchjpeg.dct as dctt

import torch
al = torch.ones(3,84,84)

x=torch.zeros(3,84,84)
x[:,:10,:42] = 1
x[:,10:21,:31] = 1
x[:,21:31,:21] = 1
x[:,31:42,:10] = 1

x2=torch.zeros(3,84,84)
x2[:,:10,42:] = 1
x2[:,10:21,31:73] = 1
x2[:,21:31,21:63] = 1
x2[:,31:42,10:52] = 1
x2[:,42:52,:42] = 1
x2[:,52:63,:31] = 1
x2[:,63:73,:21] = 1
x2[:,73:,:10] = 1

w_low = x
w_mid = x2
w_high = al-x-x2
wo_low = al-x
wo_mid = al-x2
wo_high = x+x2
#w/ low-freq:   x
#w/ mid-freq:   x2
#w/ high-freq:  x3 = al-x-x2
#w/o low-freq:  x4 = al-x
#w/o mid-freq:  x5 = al-x2
#w/o high-freq: x6 = x + x2

# Set the appropriate paths of the datasets here.
THIS_PATH = osp.dirname(__file__)
ROOT_PATH1 = osp.abspath(osp.join(THIS_PATH, '..', '..', '..'))
ROOT_PATH2 = osp.abspath(osp.join(THIS_PATH, '..', '..'))
IMAGE_PATH = osp.join(ROOT_PATH2, 'data/tiered-imagenet')
LABEL_PATH = osp.join(ROOT_PATH2, 'data/tiered-imagenet')

def buildLabelIndex(labels):
    label2inds = {}
    for idx, label in enumerate(labels):
        if label not in label2inds:
            label2inds[label] = []
        label2inds[label].append(idx)

    return label2inds


def load_data(file):
    try:
        with open(file, 'rb') as fo:
            data = pickle.load(fo)
        return data
    except:
        with open(file, 'rb') as f:
            u = pickle._Unpickler(f)
            u.encoding = 'latin1'
            data = u.load()
        return data

file_path = {'train':[os.path.join(IMAGE_PATH, 'train_images.npz'), os.path.join(IMAGE_PATH, 'train_labels.pkl')],
             'val':[os.path.join(IMAGE_PATH, 'val_images.npz'), os.path.join(IMAGE_PATH,'val_labels.pkl')],
             'test':[os.path.join(IMAGE_PATH, 'test_images.npz'), os.path.join(IMAGE_PATH, 'test_labels.pkl')]}

class tieredImageNet(data.Dataset):
    def __init__(self, setname, args, augment=False):
        assert(setname=='train' or setname=='val' or setname=='test')
        image_path = file_path[setname][0]
        label_path = file_path[setname][1]

        data_train = load_data(label_path)
        labels = data_train['label_specific']
        self.data = np.load(image_path)['images']
        label = []
        lb = -1
        self.wnids = []
        for wnid in labels:
            if wnid not in self.wnids:
                self.wnids.append(wnid)
                lb += 1
            label.append(lb)

        self.label = label
        self.num_class = len(set(label))

        if augment and setname == 'train':
            transforms_list = [
                  transforms.RandomCrop(84, padding=8),
                #   transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
                  transforms.RandomHorizontalFlip(),
                  transforms.ToTensor(),
                ]
        else:
            transforms_list = [
                  transforms.ToTensor(),
                ]

        if args.backbone_class == 'ConvNet':
            self.transform = transforms.Compose(
                transforms_list + [
                transforms.Normalize(np.array([0.485, 0.456, 0.406]),
                                     np.array([0.229, 0.224, 0.225]))
            ])
        elif args.backbone_class == 'Res12':
            self.transform = transforms.Compose(
                transforms_list + [
                transforms.Normalize(np.array([x / 255.0 for x in [120.39586422,  115.59361427, 104.54012653]]),
                                     np.array([x / 255.0 for x in [70.68188272,   68.27635443,  72.54505529]]))
            ])
        else:
            raise ValueError('Non-supported Network Types. Please Revise Data Pre-Processing Scripts.')


    def __getitem__(self, index):
        img, label = self.data[index], self.label[index]
        img = self.transform(Image.fromarray(img))
        # image = self.transform2(image)
        # cc = dct.dct_2d(image*255.-128., norm='ortho')
        # cc = dct.dct_2d(img, norm='ortho')
        # cc[:,8:,8:]=0  #only low freq

        # cc[:,:8,:8]=0
        # # # cc[:,21:,21:]=0 
        # cc[:,76:,76:]=0 #only mid freq

        # cc[:,:42,:42]=0 #only high freq

        # cc[:,:8,:8]=0      # w/o low freq
        # cc[:,8:42,8:42]=0  # w/o mid freq
        # cc[:,42:,42:]=0    # w/o high freq
        # cc[:,74:,74:]=0    # w/o high freq

        # cc[:,:21,:21]=0
        # cc[:,42:,42:]=0  #

        # dd = dct.idct_2d(cc, norm='ortho') + 128
        # dd = dct.idct_2d(cc, norm='ortho')
        # # print(dd.shape)
        # dd[dd>255]=255
        # dd[dd<0]=0
        # image = self.transform2(dd/255.)
        # img = self.transform2(dd)

        return img, label

    def __len__(self):
        return len(self.data)

class tieredImageNet_H(data.Dataset):
    
    def __init__(self, setname, args, augment=False):
        TRAIN_PATH = osp.join('/home/chenghao/DeepEMD/datasets', 'tiered_imagenet/train')
        VAL_PATH = osp.join('/home/chenghao/DeepEMD/datasets', 'tiered_imagenet/val')
        TEST_PATH = osp.join('/home/chenghao/DeepEMD/datasets', 'tiered_imagenet/test')
        if setname == 'train':
            THE_PATH = TRAIN_PATH
        elif setname == 'test':
            THE_PATH = TEST_PATH
        elif setname == 'val':
            THE_PATH = VAL_PATH
        else:
            raise ValueError('Wrong setname.')
        data = []
        label = []
        folders = [osp.join(THE_PATH, label) for label in os.listdir(THE_PATH) if
                   os.path.isdir(osp.join(THE_PATH, label))]

        csv_path = osp.join('/home/chenghao/few-shot-gnn-master/datasets/tiered_imagenet/', setname + '.csv')
        lines = [x.strip() for x in open(csv_path, 'r').readlines()][0:]

        cls_id = []
        lb = -1
        cls_dict = {}
        label2 = []
        for i in range(len(lines)):
            xx= lines[i].split(",")
            if xx[1] not in cls_id:
                cls_id.append(xx[1])
                lb += 1
            cls_dict[xx[0]]= lb

        for idx in range(len(folders)):
            this_folder = folders[idx]
            this_folder_images = os.listdir(this_folder)
            for image_path in this_folder_images:
                data.append(osp.join(this_folder, image_path))
                # label.append(idx)
                label.append(cls_dict[this_folder[-9:]])
                label2.append(idx)

        self.data = data
        self.label = label
        self.num_class = len(set(label))

        self.label2 = label2

        image_size = 84
        if augment and setname == 'train':
            transforms_list = [
                #   transforms.RandomCrop(84, padding=8),
                  transforms.RandomResizedCrop(image_size),
                  transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
                  transforms.RandomHorizontalFlip(),
                  transforms.ToTensor(),
                ]
        else:
            transforms_list = [
                  transforms.Resize(92),
                  transforms.CenterCrop(image_size),
                  transforms.ToTensor(),
                ]
        if args.backbone_class == 'ConvNet':
            self.transform = transforms.Compose(
                transforms_list + [
                transforms.Normalize(np.array([0.485, 0.456, 0.406]),
                                     np.array([0.229, 0.224, 0.225]))
            ])
        elif args.backbone_class == 'Res12':
            self.transform = transforms.Compose(
                transforms_list + [
                transforms.Normalize(np.array([x / 255.0 for x in [120.39586422,  115.59361427, 104.54012653]]),
                                     np.array([x / 255.0 for x in [70.68188272,   68.27635443,  72.54505529]]))
            ])                
        else:
            raise ValueError('Non-supported Network Types. Please Revise Data Pre-Processing Scripts.')


    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        path, label = self.data[i], self.label[i]
        image = self.transform(Image.open(path).convert('RGB'))
        return image, label


class tieredImageNet_F(data.Dataset):
    def __init__(self, setname, args, augment=False):
        assert(setname=='train' or setname=='val' or setname=='test')
        image_path = file_path[setname][0]
        label_path = file_path[setname][1]

        data_train = load_data(label_path)
        labels = data_train['labels']
        self.data = np.load(image_path)['images']
        label = []
        lb = -1
        self.wnids = []
        for wnid in labels:
            if wnid not in self.wnids:
                self.wnids.append(wnid)
                lb += 1
            label.append(lb)

        self.label = label
        self.num_class = len(set(label))

        if augment and setname == 'train':
            transforms_list = [
                  transforms.RandomCrop(84, padding=8),
                #   transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
                  transforms.RandomHorizontalFlip(),
                  transforms.ToTensor(),
                ]
        else:
            transforms_list = [
                  transforms.ToTensor(),
                ]

        self.transform = transforms.Compose(
            transforms_list)
        
        self.transform2 = transforms.Compose([
                transforms.Normalize(np.array([x / 255.0 for x in [125.3, 123.0, 113.9]]),
                                     np.array([x / 255.0 for x in [63.0, 62.1, 66.7]]))]) 

    def __getitem__(self, index):
        img, label = self.data[index], self.label[index]
        image = self.transform(Image.fromarray(img))
        cc = dct.dct_2d(image, norm='ortho')
        # cc[:,8:,8:]=0  #only low freq

        # cc[:,:8,:8]=0
        # # cc[:,21:,21:]=0 
        # cc[:,42:,42:]=0 #only mid freq

        # cc[:,:42,:42]=0 #only high freq

        # cc[:,:8,:8]=0      # w/o low freq
        # cc[:,8:42,8:42]=0  # w/o mid freq
        # cc[:,42:,42:]=0    # w/o high freq
 
        img = self.transform2(dct.idct_2d(cc,norm='ortho'))

        return img, label

    def __len__(self):
        return len(self.data)


class tieredImageNet_HF(data.Dataset):
    
    def __init__(self, setname, args, augment=False):
        TRAIN_PATH = osp.join('/home/chenghao/DeepEMD/datasets', 'tiered_imagenet/train')
        VAL_PATH = osp.join('/home/chenghao/DeepEMD/datasets', 'tiered_imagenet/val')
        TEST_PATH = osp.join('/home/chenghao/DeepEMD/datasets', 'tiered_imagenet/test')
        if setname == 'train':
            THE_PATH = TRAIN_PATH
        elif setname == 'test':
            THE_PATH = TEST_PATH
        elif setname == 'val':
            THE_PATH = VAL_PATH
        else:
            raise ValueError('Wrong setname.')
        data = []
        label = []
        folders = [osp.join(THE_PATH, label) for label in os.listdir(THE_PATH) if
                   os.path.isdir(osp.join(THE_PATH, label))]

        csv_path = osp.join('/home/chenghao/few-shot-gnn-master/datasets/tiered_imagenet/', setname + '.csv')
        lines = [x.strip() for x in open(csv_path, 'r').readlines()][0:]

        cls_id = []
        lb = -1
        cls_dict = {}
        label2 = []
        for i in range(len(lines)):
            xx= lines[i].split(",")
            if xx[1] not in cls_id:
                cls_id.append(xx[1])
                lb += 1
            cls_dict[xx[0]]= lb

        for idx in range(len(folders)):
            this_folder = folders[idx]
            this_folder_images = os.listdir(this_folder)
            for image_path in this_folder_images:
                data.append(osp.join(this_folder, image_path))
                # label.append(idx)
                label.append(cls_dict[this_folder[-9:]])
                label2.append(idx)

        self.data = data
        self.label = label
        self.num_class = len(set(label))

        self.label2 = label2

        image_size = 84
        if augment and setname == 'train':
            transforms_list = [
                #   transforms.RandomCrop(84, padding=8),
                  transforms.RandomResizedCrop(image_size),
                  transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
                  transforms.RandomHorizontalFlip(),
                  transforms.ToTensor(),
                ]
        else:
            transforms_list = [
                  transforms.Resize(92),
                  transforms.CenterCrop(image_size),
                  transforms.ToTensor(),
                ]
        
        self.transform = transforms.Compose(
            transforms_list)
        
        self.transform2 = transforms.Compose([
                transforms.Normalize(np.array([x / 255.0 for x in [120.39586422,  115.59361427, 104.54012653]]),
                                     np.array([x / 255.0 for x in [70.68188272,   68.27635443,  72.54505529]]))]) 


    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        path, label = self.data[i], self.label[i]
        image = self.transform(Image.open(path).convert('RGB'))
        # image = self.transform2(image)
        # cc = dct.dct_2d(image*255.-128., norm='ortho')
        cc = dct.dct_2d(image, norm='ortho')
        # cc[:,8:,8:]=0  #only low freq

        # cc[:,:8,:8]=0
        # cc[:,42:,42:]=0 #only mid freq

        # cc[:,:42,:42]=0 #only high freq

        # cc[:,:8,:8]=0      # w/o low freq
        # cc[:,8:42,8:42]=0  # w/o mid freq
        # cc[:,42:,42:]=0    # w/o high freq

        image = self.transform2(dct.idct_2d(cc,norm='ortho'))
    
        return image, label