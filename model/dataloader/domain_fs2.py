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

# This is for the CUB dataset
# It is notable, we assume the cub images are cropped based on the given bounding boxes
# The concept labels are based on the attribute value, which are for further use (and not used in this work)

class Domain_FS(Dataset):

    def __init__(self, setname, args, augment=False):
        # dict = {0:'sketch', 1:'quickdraw', 2:'real', 3:'painting', 4:'clipart', 5:'infograph'}
        im_size = args.orig_imsize
        self.use_im_cache = ( im_size != -1 ) # not using cache

        data = []
        label = []
        lb = -1
        self.wnids = []

        txt_path = osp.join(SPLIT_PATH, setname + '.txt')
        lines = [x.strip() for x in open(txt_path, 'r').readlines()][0:]

        for l in lines:
            context = l.split(',')
            name = context[0]
            wnid = int(context[1])
            path = osp.join(IMAGE_PATH, name)
            if wnid not in self.wnids:
                self.wnids.append(wnid)
                lb += 1

            data.append(path)
            # label.append(wnid)
            label.append(self.wnids.index(wnid))

        self.data = data
        self.label = label
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

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        data, label = self.data[i], self.label[i]
        if self.use_im_cache:
            image = self.transform(data)
        else:
            image = self.transform(Image.open(data).convert('RGB'))
        return image, label