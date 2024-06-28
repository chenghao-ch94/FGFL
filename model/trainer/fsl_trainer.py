import time
import os.path as osp
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from model.trainer.base import Trainer
from model.trainer.helpers import (
    get_dataloader, prepare_model, prepare_optimizer
)
from model.utils import (
    pprint, ensure_path,
    Averager, Timer, count_acc, one_hot,
    compute_confidence_interval,
)
from tensorboardX import SummaryWriter
from tqdm import tqdm

class FSLTrainer(Trainer):
    def __init__(self, args):
        super().__init__(args)

        self.train_loader, self.val_loader, self.test_loader = get_dataloader(args)
        self.model = prepare_model(args)
        self.optimizer, self.lr_scheduler = prepare_optimizer(self.model, args)
        self.l2 = nn.MSELoss()

    def prepare_label(self):
        args = self.args

        # prepare one-hot label
        label = torch.arange(args.way, dtype=torch.int16).repeat(args.query)
        label_aux = torch.arange(args.way, dtype=torch.int8).repeat(args.shot + args.query)
        label_q2 = torch.arange(args.way, dtype=torch.int16).repeat(args.query*2)

        label = label.type(torch.LongTensor)
        label_aux = label_aux.type(torch.LongTensor)
        label_q2 = label_q2.type(torch.LongTensor)

        if torch.cuda.is_available():
            label_s = label_s.cuda()
            label = label.cuda()
            label_aux = label_aux.cuda()
            label_q2 = label_q2.cuda()

        return label, label_aux, label_q2
  
    def train(self):
        args = self.args
        self.model.train()
        
        criterion = nn.CrossEntropyLoss()
        writer = SummaryWriter(osp.join(args.save_path,'tf'))

        # start FSL training
        label, label_aux, label_q2 = self.prepare_label()

        for epoch in range(1, args.max_epoch + 1):
            self.train_epoch += 1
            self.model.train()
            
            tl1 = Averager()
            ta = Averager()

            start_tm = time.time()
            for batch in self.train_loader:
                self.train_step += 1

                if torch.cuda.is_available():
                    data, _ = [_.cuda() for _ in batch]
                else:
                    data, _ = batch[0], batch[1]

                data_tm = time.time()
                self.dt.add(data_tm - start_tm)

                logits, logits_reg, logits_ff, logits_sm, logits_qm, loss_clr, logits_g, logits_am, logits_am2 = self.model(data) # 4-
   
                loss = criterion(logits, label)             #ori_sp
                loss2 = criterion(logits_ff, label_aux)     #ori_fq

                loss3 = criterion(logits_g, label_q2)

                # loss_am = F.softmax(logits_am, dim=1)
                # loss_am = loss_am * F.one_hot(label_aux, num_classes=args.way)
                # loss_am = loss_am.sum()/loss_am.shape[0]

                # loss_am2 = F.softmax(logits_am2, dim=1)
                # loss_am2 = loss_am2 * F.one_hot(label_aux, num_classes=args.way)
                # loss_am2 = loss_am2.sum()/loss_am2.shape[0]
            
                # loss_sm = F.softmax(logits_sm, dim=1)
                # loss_sm = loss_sm * F.one_hot(label, num_classes=args.way)
                # loss_sm = loss_sm.sum()/loss_sm.shape[0]

                # loss_qm = F.softmax(logits_qm, dim=1)
                # loss_qm = loss_qm * F.one_hot(label, num_classes=args.way)
                # loss_qm = loss_qm.sum()/loss_qm.shape[0]

                loss_am = criterion(logits_am, label_aux)
                loss_am2 = criterion(logits_am2, label_aux)
                loss_sm = criterion(logits_sm, label)
                loss_qm = criterion(logits_qm, label)

                total_loss = loss + 1.0*loss2 + 0.1*(loss_am + loss_am2 + loss_sm + loss_qm) + 0.01*loss3 + 0.01*loss_clr

                if logits_reg is not None:
                    total_loss = total_loss + args.balance * F.cross_entropy(logits_reg, label_aux)

                forward_tm = time.time()
                self.ft.add(forward_tm - data_tm)
                acc = count_acc(logits, label)
                acc_f = count_acc(logits_ff, label_aux)
                acc_g = count_acc(logits_g, label_q2)
            
                writer.add_scalar('data/loss_sp', float(loss.item()), epoch)
                writer.add_scalar('data/loss_fq', float(loss2.item()), epoch)
                writer.add_scalar('data/loss_aug', float(loss3.item()), epoch)

                writer.add_scalar('data/acc_sp', float(acc), epoch)
                writer.add_scalar('data/acc_fq', float(acc_f), epoch)
                writer.add_scalar('data/acc_aug', float(acc_g), epoch)

                writer.add_scalar('data/loss_am', float(loss_am.item()), epoch)
                writer.add_scalar('data/loss_sm', float(loss_sm.item()), epoch)
                writer.add_scalar('data/loss_qm', float(loss_qm.item()), epoch)

                tl1.add(total_loss.item())
                ta.add(acc)

                self.optimizer.zero_grad()
                total_loss.backward()

                backward_tm = time.time()
                self.bt.add(backward_tm - forward_tm)

                self.optimizer.step()
                optimizer_tm = time.time()
                self.ot.add(optimizer_tm - backward_tm)    

                # refresh start_tm
                start_tm = time.time()

            self.lr_scheduler.step()
            self.try_evaluate(epoch)
            # torch.cuda.synchronize()

            print('ETA:{}/{}'.format(
                    self.timer.measure(),
                    self.timer.measure(self.train_epoch / args.max_epoch))
            )

        writer.close()
        torch.save(self.trlog, osp.join(args.save_path, 'trlog'))
        self.save_model('epoch-last')

    def evaluate(self, data_loader, epoch):
        # restore model args
        args = self.args
        # evaluation mode
        self.model.eval()

        record = np.zeros((args.num_eval_episodes, 2)) # loss and acc

        label_tr = torch.arange(args.eval_way, dtype=torch.int16).repeat(args.eval_shot)
        label_tr = label_tr.type(torch.LongTensor)
        if torch.cuda.is_available():
            label_tr = label_tr.cuda()                
        label = torch.arange(args.eval_way, dtype=torch.int16).repeat(args.eval_query)
        label = label.type(torch.LongTensor)
        if torch.cuda.is_available():
            label = label.cuda()

        label_aux = torch.arange(args.eval_way, dtype=torch.int16).repeat(args.eval_shot + args.eval_query)
        label_aux = label_aux.type(torch.LongTensor)
        if torch.cuda.is_available():
            label_aux = label_aux.cuda()

        print('best epoch {}, best val acc={:.3f} + {:.3f}'.format(
                self.trlog['max_acc_epoch'],
                self.trlog['max_acc'],
                self.trlog['max_acc_interval']))

        with torch.no_grad():
            for i, batch in enumerate(data_loader, 1):
                if torch.cuda.is_available():
                    data, _ = [_.cuda() for _ in batch]
                else:
                    data = batch[0]

                logits_s = self.model(data)
                acc_s = count_acc(logits_s, label)
                loss = F.cross_entropy(logits_s, label)
                record[i-1, 0] = loss.item()
                record[i-1, 1] = acc_s

        assert(i == record.shape[0])
        vl, _ = compute_confidence_interval(record[:,0])
        va, vap = compute_confidence_interval(record[:,1])

        self.trlog['test_acc'] = self.trlog['max_acc']
        self.trlog['test_acc_interval'] = self.trlog['max_acc_interval']

        print('Eval val oriacc={:.3f} + {:.3f}'.format(va, vap))

        # train mode
        self.model.train()
        return vl, va, vap

    def evaluate_test(self):
        # restore model args
        args = self.args
        # evaluation mode
        self.model.load_state_dict(torch.load(osp.join(self.args.save_path, 'max_acc.pth'))['params'])
        self.model.eval()
        # self.model.train()
        record = np.zeros((600, 2)) # loss and acc
        label_tr = torch.arange(args.eval_way, dtype=torch.int16).repeat(args.eval_shot)
        label_tr = label_tr.type(torch.LongTensor)
        if torch.cuda.is_available():
            label_tr = label_tr.cuda()   
        label = torch.arange(args.eval_way, dtype=torch.int16).repeat(args.eval_query)
        label = label.type(torch.LongTensor)
        if torch.cuda.is_available():
            label = label.cuda()
        label_aux = torch.arange(args.eval_way, dtype=torch.int16).repeat(args.eval_shot + args.eval_query)
        label_aux = label_aux.type(torch.LongTensor)
        if torch.cuda.is_available():
            label_aux = label_aux.cuda()
        print('best epoch {}, best val acc={:.4f} + {:.4f}'.format(
                self.trlog['max_acc_epoch'],
                self.trlog['max_acc'],
                self.trlog['max_acc_interval']))

        print(args.save_path)
        writer = SummaryWriter(osp.join(args.save_path,'tf'))      

        with torch.no_grad():
            for i, batch in tqdm(enumerate(self.test_loader, 1)):
                if torch.cuda.is_available():
                    data, _ = [_.cuda() for _ in batch]
                else:
                    data = batch[0]

                logits = self.model(data)
                loss = F.cross_entropy(logits, label)
                acc = count_acc(logits, label)
                record[i-1, 0] = loss.item()
                record[i-1, 1] = acc

        assert(i == record.shape[0])
        vl, _ = compute_confidence_interval(record[:,0])
        va, vap = compute_confidence_interval(record[:,1])

        self.trlog['test_acc'] = va
        self.trlog['test_acc_interval'] = vap
        self.trlog['test_loss'] = vl

        print('Test acc={:.4f} + {:.4f}\n'.format(
                self.trlog['test_acc'],
                self.trlog['test_acc_interval']))

        writer.close()
        return vl, va, vap

    def final_record(self):
        # save the best performance in a txt file
        
        with open(osp.join(self.args.save_path, '{}+{}'.format(self.trlog['test_acc'], self.trlog['test_acc_interval'])), 'w') as f:
            f.write('best epoch {}, best val acc={:.4f} + {:.4f}\n'.format(
                self.trlog['max_acc_epoch'],
                self.trlog['max_acc'],
                self.trlog['max_acc_interval']))
            f.write('Test acc={:.4f} + {:.4f}\n'.format(
                self.trlog['test_acc'],
                self.trlog['test_acc_interval']))   