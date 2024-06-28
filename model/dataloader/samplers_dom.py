import torch
import numpy as np
import random


class CategoriesSampler():

    def __init__(self, label, n_batch, n_cls, n_per):
        self.n_batch = n_batch# the number of iterations in the dataloader
        self.n_cls = n_cls
        self.n_per = n_per  # 5+1 or 5+15
    
        label = np.array(label)#all data label
        self.m_ind = []#the data index of each class
        label_u = np.unique(label)

        for i in range(len(label_u)):
            ind = np.argwhere(label == label_u[i]).reshape(-1)# all data index of this class
            ind = torch.from_numpy(ind)
            self.m_ind.append(ind)

    def __len__(self):
        return self.n_batch
    
    def __iter__(self):
        for i_batch in range(self.n_batch):
            batch = []
            classes = torch.randperm(len(self.m_ind))[:self.n_cls]
            for c in classes:
                l = self.m_ind[c]
                pos = torch.randperm(len(l))[:self.n_per]
                batch.append(l[pos])
            batch = torch.stack(batch).t().reshape(-1)
            yield batch