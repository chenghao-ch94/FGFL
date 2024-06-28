import torch
import numpy as np
import random


class CategoriesSampler():

    def __init__(self, label, n_batch, n_cls, n_per, domain, d_id):
        self.n_batch = n_batch# the number of iterations in the dataloader
        self.n_cls = n_cls
        self.n_per = n_per  # 5+1 or 5+15
        self.domain = domain
        
        if isinstance(d_id,int):
            self.d_id = d_id
        else:
            self.d_id = np.array(d_id)

        label = np.array(label)#all data label
        self.domain = np.array(self.domain)
        self.m_ind = []#the data index of each class
        assert len(label)==len(self.domain)
        
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
            classes = torch.randperm(len(self.m_ind))[:self.n_cls]#random sample num_class indexs,e.g. 5
            # print(classes)
            for c in classes:
                l = self.m_ind[c]#all data indexs of this class
                dl = self.domain[l]#all domain label of this class

                # if len(self.d_id)==1:      # test set with only one domain, same setting with original few-shot
                if isinstance(self.d_id, int):      # test set with only one domain, same setting with original few-shot
                    pos = torch.randperm(len(l))[:self.n_per] #sample n_per data index of this class
                    batch.append(l[pos])

                else:                # train and val set with five domains
                    assert(len(self.d_id)==5)
                    if self.n_per == 16:   # 1-shot, 1 support for 1 random domain and 3 query for each domain
                        in_d = [j for j in range(len(self.d_id))]
                        random.shuffle(in_d)
                        d_n = self.d_id[in_d]
                        domain_s = d_n[0]            # the selected domain for 1 support sample
                        # print('domain label',dl)
                        # print('choose', domain_s)
                        l2 = np.argwhere(dl==domain_s)
                        assert len(l2)>0
                        pos1 = torch.randperm(len(l2))[:1]
                        # batch.append(l[pos])
                       
                    elif self.n_per == 20: # 5-shot, 1 support and 3 query for each domain
                        temp=[]
                        for i in range(len(self.d_id)):
                            l2 = np.argwhere(dl==self.d_id[i])
                            pos1 = torch.randperm(len(l2))[:1] #sample 1 support index of each domain

                            if len(temp) == 0:
                                temp = pos1
                            else:
                                temp = torch.cat([temp, pos1],dim=0)
                            # batch.append(l[pos1])
                        pos1 = temp
                    
                    for i in range(len(self.d_id)):
                        l2 = np.argwhere(dl==self.d_id[i])  #all data indexs of this class with certain domain
                        pos2 = torch.randperm(len(l2))[:3] #sample 1 support and 3 query data index of this class
                        pos1 = torch.cat([pos1,pos2],dim=0)
                        # batch.append(l[pos])

                    # print(len(pos1))
                    batch.append(l[pos1])
                    # print(batch)

            batch = torch.stack(batch).t().reshape(-1)
            # print(batch)
            # .t() transpose,
            # due to it, the label is in the sequence of abcdabcdabcd form after reshape,
            # instead of aaaabbbbccccdddd
            yield batch

class CategoriesSampler_h2c():
    
    def __init__(self, label_h, label_c, n_batch, n_cls, n_per):
        self.n_batch = n_batch# the number of iterations in the dataloader
        self.n_cls = n_cls
        self.n_per = n_per

        self.label_c = np.array(label_c)#all data label
        label_h = np.array(label_h)
        self.m_ind = []#the data index of each class
        for i in range(max(label_h) + 1):
            ind = np.argwhere(label_h == i).reshape(-1)# all data index of this class
            ind = torch.from_numpy(ind)
            self.m_ind.append(ind)

    def __len__(self):
        return self.n_batch
    
    def __iter__(self):
        for i_batch in range(self.n_batch):
            batch = []
            classes = torch.randperm(len(self.m_ind))[:1]#random select 1 high-class
            cls_idx = self.m_ind[classes] #all data indexes of this high-class

            sub_cls_idx = self.label_c[cls_idx] # all sub-class labels of this high-class
            sub_cls = np.unique(sub_cls_idx)    # all sub-class names
            sub_classes = torch.randperm(len(sub_cls))[:self.n_cls]

            for c in sub_classes:
                m = np.argwhere(sub_cls_idx == sub_cls[c])
                pos = torch.randperm(len(m))[:self.n_per] #sample n_per data index of this class
                batch.append(cls_idx[torch.from_numpy(m[pos])].reshape(-1))

            batch = torch.stack(batch).t().reshape(-1)
            # print(batch)
            # .t() transpose,
            # due to it, the label is in the sequence of abcdabcdabcd form after reshape,
            # instead of aaaabbbbccccdddd
            yield batch
