import torch
import tqdm
from torch.utils.data import DataLoader
import torchjpeg.dct as dctt
from model.dataloader.tiered_imagenet import tieredImageNet as Dataset
# from model.dataloader.mini_imagenet import MiniImageNet as Dataset


psum = torch.zeros(3,84,84)
psum_sq = torch.zeros(3,84,84)

trainset = Dataset('train', args=[], augment=True)
train_loader = DataLoader(dataset=trainset, batch_size=1, shuffle=False, num_workers=8, pin_memory=True)

c = 0
for i, batch in enumerate(train_loader, 1):
    
    inputs, _ = batch
    inputs = dctt.images_to_batch(inputs).squeeze()

    psum += inputs
    psum_sq += inputs ** 2

    print(psum)
    print(psum_sq)

    c += 1

count = c

# mean and std
total_mean = psum / count
total_var  = (psum_sq / count) - (total_mean ** 2)
total_std  = torch.sqrt(total_var)

# output
print('mean: '  + str(total_mean))
print('std:  '  + str(total_std))