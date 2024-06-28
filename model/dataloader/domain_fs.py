import os.path as osp
import PIL
from PIL import Image

import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms

THIS_PATH = osp.dirname(__file__)
ROOT_PATH1 = osp.abspath(osp.join(THIS_PATH, '..', '..', '..'))
ROOT_PATH2 = osp.abspath(osp.join(THIS_PATH, '..', '..'))
# IMAGE_PATH = osp.join(ROOT_PATH1, 'FEAT/data/cub')
IMAGE_PATH = '/home/chenghao/domainnet/'
SPLIT_PATH = osp.join(ROOT_PATH2, 'data/domainnet/split')
CACHE_PATH = osp.join(ROOT_PATH2, '.cache/')

import torch_dct as dct

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
# This is for the CUB dataset
# It is notable, we assume the cub images are cropped based on the given bounding boxes
# The concept labels are based on the attribute value, which are for further use (and not used in this work)

class Domain_FS(Dataset):

    def __init__(self, d_ids, setname, args, augment=False):
        dict = {0:'sketch', 1:'quickdraw', 2:'real', 3:'painting', 4:'clipart', 5:'infograph'}
        im_size = args.orig_imsize
        self.use_im_cache = ( im_size != -1 ) # not using cache

        data = []
        label = []
        label_d = []
        self.wnids = []
        lb=-1

        if isinstance(d_ids,int):
            
            d_id = d_ids
            txt_path = osp.join(SPLIT_PATH, dict[d_id] + '_' + setname + '.txt')
            lines = [x.strip() for x in open(txt_path, 'r').readlines()][0:]

            for l in lines:
                context = l.split(',')
                name = context[0]
                wnid = int(context[1])
                # did = int(context[2])
                path = osp.join(IMAGE_PATH, name)
                if wnid not in self.wnids:
                    self.wnids.append(wnid)
                    lb += 1

                data.append(path)
                label.append(wnid)
                label_d.append(d_id)

        else:
            for doid in range(len(d_ids)):
            
                d_id = d_ids[doid]
                txt_path = osp.join(SPLIT_PATH, dict[d_id] + '_' + setname + '.txt')
                lines = [x.strip() for x in open(txt_path, 'r').readlines()][0:]

                for l in lines:
                    context = l.split(',')
                    name = context[0]
                    wnid = int(context[1])
                    # did = int(context[2])
                    path = osp.join(IMAGE_PATH, name)
                    if wnid not in self.wnids:
                        self.wnids.append(wnid)
                        lb += 1

                    data.append(path)
                    label.append(wnid)
                    label_d.append(d_id)
        
        # cache_path = osp.join( CACHE_PATH, "{}.{}.{}.pt".format(self.__class__.__name__, setname, im_size) )
        # self.use_im_cache = ( im_size != -1 ) # not using cache
        # if self.use_im_cache:
        #     if not osp.exists(cache_path):
        #         print('* Cache miss... Preprocessing {}...'.format(setname))
        #         resize_ = identity if im_size < 0 else transforms.Resize(im_size)
        #         data, label = self.parse_csv(txt_path)
        #         self.data = [ resize_(Image.open(path).convert('RGB')) for path in data ]
        #         self.label = label
        #         print('* Dump cache from {}'.format(cache_path))
        #         torch.save({'data': self.data, 'label': self.label }, cache_path)
        #     else:
        #         print('* Load cache from {}'.format(cache_path))
        #         cache = torch.load(cache_path)
        #         self.data  = cache['data']
        #         self.label = cache['label']
        # else:
        #     self.data, self.label = self.parse_csv(txt_path)
        
        self.data = data
        self.label = label
        self.domain = label_d
        self.num_class = np.unique(np.array(self.label)).shape[0]
        image_size = 84
        
        if augment and setname == 'train':
            transforms_list = [
                  transforms.Resize(92),
                  transforms.RandomCrop(image_size),                
                #   transforms.RandomResizedCrop(image_size),
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

        # Transformation
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
        elif args.backbone_class == 'Res18':
            self.transform = transforms.Compose(
                transforms_list + [
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
            ])            
        elif args.backbone_class == 'WRN':
            self.transform = transforms.Compose(
                transforms_list + [
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
            ])         
        else:
            raise ValueError('Non-supported Network Types. Please Revise Data Pre-Processing Scripts.')

    # def parse_csv(self, txt_path):
    #     data = []
    #     label = []
    #     lb = -1
    #     self.wnids = []
    #     lines = [x.strip() for x in open(txt_path, 'r').readlines()][1:]

    #     for l in lines:
    #         context = l.split(',')
    #         name = context[0] 
    #         wnid = context[1]
    #         path = osp.join(IMAGE_PATH, name)
    #         if wnid not in self.wnids:
    #             self.wnids.append(wnid)
    #             lb += 1
                
    #         data.append(path)
    #         label.append(lb)

    #     return data, label

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        data, label = self.data[i], self.label[i]
        if self.use_im_cache:
            image = self.transform(data)
        else:
            image = self.transform(Image.open(data).convert('RGB'))

        # return image, label
        return image, self.domain[i]

class Domain_FS_F(Dataset):
    
    def __init__(self, d_ids, setname, args, augment=False):
        dict = {0:'sketch', 1:'quickdraw', 2:'real', 3:'painting', 4:'clipart', 5:'infograph'}
        im_size = args.orig_imsize
        self.use_im_cache = ( im_size != -1 ) # not using cache

        data = []
        label = []
        label_d = []
        self.wnids = []
        lb=-1

        if isinstance(d_ids,int):
            
            d_id = d_ids
            txt_path = osp.join(SPLIT_PATH, dict[d_id] + '_' + setname + '.txt')
            lines = [x.strip() for x in open(txt_path, 'r').readlines()][0:]

            for l in lines:
                context = l.split(',')
                name = context[0]
                wnid = int(context[1])
                # did = int(context[2])
                path = osp.join(IMAGE_PATH, name)
                if wnid not in self.wnids:
                    self.wnids.append(wnid)
                    lb += 1

                data.append(path)
                label.append(wnid)
                label_d.append(d_id)

        else:
            for doid in range(len(d_ids)):
            
                d_id = d_ids[doid]
                txt_path = osp.join(SPLIT_PATH, dict[d_id] + '_' + setname + '.txt')
                lines = [x.strip() for x in open(txt_path, 'r').readlines()][0:]

                for l in lines:
                    context = l.split(',')
                    name = context[0]
                    wnid = int(context[1])
                    # did = int(context[2])
                    path = osp.join(IMAGE_PATH, name)
                    if wnid not in self.wnids:
                        self.wnids.append(wnid)
                        lb += 1

                    data.append(path)
                    label.append(wnid)
                    label_d.append(d_id)
                    print(d_id)
        
        self.data = data
        self.label = label
        self.domain = label_d
        self.num_class = np.unique(np.array(self.label)).shape[0]
        image_size = 84
        
        print(self.domain)

        if augment and setname == 'train':
            transforms_list = [
                  transforms.Resize(92),
                  transforms.RandomCrop(image_size),                
                #   transforms.RandomResizedCrop(image_size),
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
        data, label = self.data[i], self.label[i]
        if self.use_im_cache:
            image = self.transform(data)
        else:
            image = self.transform(Image.open(data).convert('RGB'))

        # image = self.transform2(image)
        cc = dct.dct_2d(image, norm='ortho')
        # cc[:,8:,8:]=0  #only low freq

        # cc[:,:8,:8]=0
        # cc[:,42:,42:]=0 #only mid freq

        # cc[:,:42,:42]=0 #only high freq

        # cc[:,:8,:8]=0      # w/o low freq
        # cc[:,8:42,8:42]=0  # w/o mid freq
        # cc[:,42:,42:]=0    # w/o high freq

        # cc[:,:21,:21]=0
        # cc[:,42:,42:]=0  #
        # cc = cc*w_mid
        image = dct.idct_2d(cc, norm='ortho')
        # image = self.transform2(dct.idct_2d(cc))
        
        label_d = self.domain[i]

        # return image, label
        return image, label_d