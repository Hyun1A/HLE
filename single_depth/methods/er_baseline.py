# When we make a new one, we should inherit the Finetune class.
import os, sys
import logging
import copy
from copy import deepcopy
import time
import datetime
import json


import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch import optim

from utils.data_loader import ImageDataset, StreamDataset, MemoryDataset, cutmix_data, get_statistics
from utils.train_utils import select_model, select_optimizer, select_scheduler

logger = logging.getLogger()
#writer = SummaryWriter("tensorboard")


def cycle(iterable):
    # iterate with shuffling
    while True:
        for i in iterable:
            yield i


class ER:
    def __init__(
            self, criterion, device, train_transform, test_transform, n_classes, n_classes_sup, n_classes_sub, writer, **kwargs
    ):
        
        self.root = kwargs["root"]
        self.exp_name = kwargs["exp_name"]
        self.tm = kwargs["tm"]
        self.seed = kwargs["rnd_seed"]
        self.mode = kwargs["mode"]
        self.cur_iter = 0
        self.n_count_num = 0
        self.writer = writer
        
        self.num_learned_class = 0
        self.num_learned_class_sup = 0
        self.num_learned_class_sub = 0
        self.sup_param_idx = 0
        self.sub_param_idx = 1
        
        self.num_learning_class = 1
        self.n_classes = n_classes
        self.n_classes_sup = n_classes_sup
        self.n_classes_sub = n_classes_sub
        
        self.exposed_classes = []
        self.check_stream = 0
        self.exposed_classes_sup = []
        self.exposed_classes_sub = []
        self.corresponding_super = []

        
        self.seen = 0
        self.topk = kwargs["topk"]

        self.device = device
        self.dataset = kwargs["dataset"]
        self.model_name = kwargs["model_name"]
        self.opt_name = kwargs["opt_name"]
        self.sched_name = kwargs["sched_name"]
        if self.sched_name == "default":
            self.sched_name = 'exp_reset'
        self.lr = kwargs["lr"]

        self.train_transform = train_transform
        self.cutmix = "cutmix" in kwargs["transforms"]
        self.test_transform = test_transform

        self.memory_size = kwargs["memory_size"]
        self.memory_size_total = kwargs["memory_size"]
        
        self.data_dir = kwargs["data_dir"]

        self.online_iter = kwargs["online_iter"]
        self.batch_size = kwargs["batchsize"]
        self.temp_batchsize = kwargs["temp_batchsize"]
        if self.temp_batchsize is None:
            self.temp_batchsize = self.batch_size//2
        if self.temp_batchsize > self.batch_size:
            self.temp_batchsize = self.batch_size
        #self.memory_size -= self.temp_batchsize

        self.gpu_transform = kwargs["gpu_transform"]
        self.use_amp = kwargs["use_amp"]
        if self.use_amp:
            self.scaler = torch.cuda.amp.GradScaler()
            
        self.model = select_model(self.model_name, self.dataset, 1).to(self.device)
        self.optimizer = select_optimizer(self.opt_name, self.lr, self.model)
        
        if 'imagenet' in self.dataset:
            self.lr_gamma = 0.9999
        else:
            self.lr_gamma = 0.9999
        self.scheduler = select_scheduler(self.sched_name, self.optimizer, self.lr_gamma)

        self.criterion = criterion.to(self.device)
        self.memory = MemoryDataset(self.root, self.dataset, self.train_transform, self.exposed_classes,
                                    test_transform=self.test_transform, data_dir=self.data_dir, device=self.device,
                                    transform_on_gpu=self.gpu_transform, n_classes_sup=self.n_classes_sup)
                
        self.temp_batch = []
        self.num_updates = 0
        self.train_count = 0
        self.batch_size = kwargs["batchsize"]

        self.start_time = time.time()
        num_samples = {'cifar10': 50000, 'cifar100': 50000, 'tinyimagenet': 100000, 'imagenet': 1281167, 'imagenet_subset': 128741*2, 'imagenet_subset_sub_shuffle': 128741*2, 'cifar100_hier_setup1':100000,  'cifar100_hier_setup2':50000, 'cifar100_hier_setup3':50000, 'stanford_car_setup1':8144*2, 'imagenet_subset_setup2': 128741, 'stanford_car_setup2':8144}
        self.total_samples = num_samples[self.dataset]
    def online_step(self, sample, sample_num, n_worker):
        #print('exposed_classes:', self.exposed_classes)
        self.n_count_num +=1
        
        
        if sample['klass'] not in self.exposed_classes:
            self.add_new_class(sample['klass'], sample['hierarchy'], sample['klass_sup'])

        self.temp_batch.append(sample)
        self.num_updates += self.online_iter

        if len(self.temp_batch) == self.temp_batchsize:
            if self.check_stream == 0:
                train_loss_sup, train_acc_sup, train_loss_sub, train_acc_sub = \
                                                          self.online_train(self.temp_batch, self.batch_size, n_worker, \
                                                          iterations=int(self.num_updates), stream_batch_size=self.temp_batchsize)
            
                print('super class')
                self.report_training(sample_num, train_loss_sup, train_acc_sup)
                print('sub class')
                self.report_training(sample_num, train_loss_sub, train_acc_sub)            

            
            for stored_sample in self.temp_batch:
                self.update_memory(stored_sample)
            self.temp_batch = []
            self.num_updates -= int(self.num_updates)

    def add_new_class(self, class_name, hierarchy, class_name_sup=None, class_name_sub=None):
        self.exposed_classes.append(class_name)
        self.num_learned_class = len(self.exposed_classes)
        
        if hierarchy == 0:
            self.num_learned_class_sup += 1
            self.exposed_classes_sup.append(class_name)
            
            prev_weight = copy.deepcopy(self.model.fc_sup.weight.data)
            self.model.fc_sup = nn.Linear(self.model.fc_sup.in_features, self.num_learned_class_sup).to(self.device)


            with torch.no_grad():
                if self.num_learned_class_sup > 1:
                    self.model.fc_sup.weight[:self.num_learned_class_sup - 1] = prev_weight
            for param in self.optimizer.param_groups[self.sup_param_idx+1]['params']:
                if param in self.optimizer.state.keys():
                    del self.optimizer.state[param]
            del self.optimizer.param_groups[self.sup_param_idx+1]
            self.optimizer.add_param_group({'params': self.model.fc_sup.parameters()})
            self.memory.add_new_class(cls_list=self.exposed_classes)
            if self.sup_param_idx == 0:
                self.sup_param_idx = (self.sup_param_idx+1)%2
                self.sub_param_idx = (self.sub_param_idx+1)%2
                
        else:
            self.num_learned_class_sub += 1 
            self.exposed_classes_sub.append(class_name)
            self.corresponding_super.append(self.exposed_classes.index(class_name_sup))
            
            
            prev_weight = copy.deepcopy(self.model.fc_sub.weight.data)
            self.model.fc_sub = nn.Linear(self.model.fc_sub.in_features, self.num_learned_class_sub).to(self.device)


            with torch.no_grad():
                if self.num_learned_class_sub > 1:
                    self.model.fc_sub.weight[:self.num_learned_class_sub - 1] = prev_weight
            for param in self.optimizer.param_groups[self.sub_param_idx+1]['params']:
                if param in self.optimizer.state.keys():
                    del self.optimizer.state[param]
            del self.optimizer.param_groups[self.sub_param_idx+1]
            self.optimizer.add_param_group({'params': self.model.fc_sub.parameters()})
            self.memory.add_new_class(cls_list=self.exposed_classes)
            if self.sub_param_idx == 0:
                self.sup_param_idx = (self.sup_param_idx+1)%2
                self.sub_param_idx = (self.sub_param_idx+1)%2
        
        
        if 'reset' in self.sched_name:
            self.update_schedule(reset=True)


            
    def online_train(self, sample, batch_size, n_worker, iterations=1, stream_batch_size=1):
        total_loss_sup, correct_sup, num_data_sup = 0.0, 0.0, 0.0
        total_loss_sub, correct_sub, num_data_sub = 0.0, 0.0, 0.0
        
        if stream_batch_size > 0:
            sample_dataset = StreamDataset(self.root, sample, dataset=self.dataset, transform=self.train_transform,
                                           cls_list=self.exposed_classes, data_dir=self.data_dir, device=self.device,
                                           transform_on_gpu=self.gpu_transform, n_classes_sup=self.n_classes_sup)
            
        if len(self.memory) > 0 and batch_size - stream_batch_size > 0:
            memory_batch_size = min(len(self.memory), batch_size - stream_batch_size)

            
            
        for i in range(iterations):
            self.model.train()
            x = []
            y = []
            hierarchy = []
            
            if stream_batch_size > 0:
                stream_data = sample_dataset.get_data()
                x.append(stream_data['image'])
                y.append(stream_data['label'])
                hierarchy.append(stream_data['hierarchy'])
                
            if len(self.memory) > 0 and batch_size - stream_batch_size > 0:
                memory_data = self.memory.get_batch(memory_batch_size)
                x.append(memory_data['image'])
                y.append(memory_data['label'])
                hierarchy.append(memory_data['hierarchy'])
                
            x = torch.cat(x)
            y = torch.cat(y)
            hierarchy = torch.cat(hierarchy)

            
            x = x.to(self.device)
            y = y.to(self.device)
            hierarchy = hierarchy.to(self.device)

            self.optimizer.zero_grad()

            logit_sup, logit_sub, loss_sup, loss_sub = self.model_forward(x,y, hierarchy)

            loss = (loss_sup + loss_sub)/x.size(0)

            
            
            
            if self.use_amp:
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                self.optimizer.step()

            self.update_schedule()

            _, preds_sup = logit_sup.topk(self.topk, 1, True, True)
            _, preds_sub = logit_sub.topk(self.topk, 1, True, True)
                                    
            idx_sup = (hierarchy == 0)
            idx_sub = (hierarchy == 1)             
            
            total_loss_sup += loss_sup.item()
            correct_sup += torch.sum(preds_sup[idx_sup] == y[idx_sup].unsqueeze(1)).item()
            num_data_sup += y[idx_sup].size(0)
            
            total_loss_sub += loss_sub.item()
            correct_sub += torch.sum(preds_sub[idx_sub] == y[idx_sub].unsqueeze(1)).item()
            num_data_sub += y[idx_sub].size(0)                 

        
        train_loss_sup = total_loss_sup / num_data_sup if num_data_sup != 0 else 0.
        train_acc_sup = correct_sup / num_data_sup if num_data_sup != 0 else 0.
            
        train_loss_sub = total_loss_sub / num_data_sub if num_data_sub != 0 else 0.
        train_acc_sub = correct_sub / num_data_sub if num_data_sub != 0 else 0.            
            
            
        return train_loss_sup, train_acc_sup, train_loss_sub, train_acc_sub
    
    def model_forward(self, x, y, hierarchy):
        do_cutmix = self.cutmix and np.random.rand(1) < 0.5
        
        idx_sup = (hierarchy == 0)
        idx_sub = (hierarchy == 1)
        
        num_idx_sup = torch.sum(idx_sup.type(torch.float))
        num_idx_sub = torch.sum(idx_sub.type(torch.float))   
            
        if (torch.sum(idx_sup.type(torch.float)) == 1) or (torch.sum(idx_sub.type(torch.float)) == 1):
            do_cutmix=False
        
        #print(do_cutmix)
        #print(self.model)
        #print(y)
        #print(hierarchy)
        
        
        if do_cutmix:
            if self.use_amp:
                with torch.cuda.amp.autocast():
                    labels_a = deepcopy(y)
                    labels_b = deepcopy(y)

                    x, labels_a, labels_b, lam = cutmix_data(x=x, y=y, alpha=1.0)

                    x_sup, labels_a_sup, labels_b_sup, lam_sup = cutmix_data(x=x[idx_sup], y=y[idx_sup], alpha=1.0)
                    x_sub, labels_a_sub, labels_b_sub, lam_sub = cutmix_data(x=x[idx_sub], y=y[idx_sub], alpha=1.0)

                    x[idx_sup] = x_sup
                    x[idx_sub] = x_sub

                    labels_a[idx_sup] = labels_a_sup
                    labels_a[idx_sub] = labels_a_sub

                    labels_b[idx_sup] = labels_b_sup
                    labels_b[idx_sub] = labels_b_sub

                    lam = (num_idx_sup*lam_sup + num_idx_sub*lam_sub) / (num_idx_sup+num_idx_sub)
                    
                        
                    logit_sup, logit_sub = self.model(x)
                    #loss_sup = self.criterion(logit_sup[idx_sup], y[idx_sup]) \
                    #    if torch.sum(idx_sup.type(torch.float)) != 0 else torch.tensor([0.]).to(self.device)

                    #loss_sub = self.criterion(logit_sub[idx_sub], y[idx_sub]) \
                    #    if torch.sum(idx_sub.type(torch.float)) != 0 else torch.tensor([0.]).to(self.device)                        
                        
                    loss_sup = lam*self.criterion(logit_sup[idx_sup], labels_a[idx_sup]) + (1 - lam) * self.criterion(logit_sup[idx_sup], labels_b[idx_sup]) \
                        if torch.sum(idx_sup.type(torch.float)) != 0 else torch.tensor([0.]).to(self.device)

                    loss_sub = lam*self.criterion(logit_sub[idx_sub], labels_a[idx_sub]) + (1 - lam) * self.criterion(logit_sub[idx_sub], labels_b[idx_sub]) \
                        if torch.sum(idx_sub.type(torch.float)) != 0 else torch.tensor([0.]).to(self.device)                      
                        
                                                
                        
                        
                        

        else:            
            if self.use_amp:
                with torch.cuda.amp.autocast():
                    
                    
                    logit_sup, logit_sub = self.model(x)
                    loss_sup = self.criterion(logit_sup[idx_sup], y[idx_sup]) \
                        if torch.sum(idx_sup.type(torch.float)) != 0 else torch.tensor([0.]).to(self.device)

                    loss_sub = self.criterion(logit_sub[idx_sub], y[idx_sub]) \
                        if torch.sum(idx_sub.type(torch.float)) != 0 else torch.tensor([0.]).to(self.device)


        return logit_sup, logit_sub, loss_sup, loss_sub

    def report_training(self, sample_num, train_loss, train_acc):
        self.writer.add_scalar(f"train/loss", train_loss, sample_num)
        self.writer.add_scalar(f"train/acc", train_acc, sample_num)
        logger.info(
            f"Train | Sample # {sample_num} | train_loss {train_loss:.4f} | train_acc {train_acc:.4f} | "
            f"lr {self.optimizer.param_groups[0]['lr']:.6f} | "
            f"running_time {datetime.timedelta(seconds=int(time.time() - self.start_time))} | "
            f"ETA {datetime.timedelta(seconds=int((time.time() - self.start_time) * (self.total_samples-sample_num) / sample_num))}"
        )

    def report_test(self, sample_num, avg_loss, avg_acc):
        self.writer.add_scalar(f"test/loss", avg_loss, sample_num)
        self.writer.add_scalar(f"test/acc", avg_acc, sample_num)
        logger.info(
            f"Test | Sample # {sample_num} | test_loss {avg_loss:.4f} | test_acc {avg_acc:.4f} | "
        )

    def update_memory(self, sample):
        self.reservoir_memory(sample)

    def update_schedule(self, reset=False):
        if reset:
            self.scheduler = select_scheduler(self.sched_name, self.optimizer, self.lr_gamma)
            for param_group in self.optimizer.param_groups:
                param_group["lr"] = self.lr
        else:
            self.scheduler.step()

    def online_evaluate(self, test_list, sample_num, batch_size, n_worker, end_task=False):
        test_df = pd.DataFrame(test_list)
        
        exp_test_df = test_df[test_df['klass'].isin(self.exposed_classes)]
        
        #print('exp_test_df:', exp_test_df)
        #print('self.n_classes_sup:', self.n_classes_sup)
        
        test_dataset = ImageDataset(
            self.root,
            exp_test_df,
            dataset=self.dataset,
            transform=self.test_transform,
            cls_list=self.exposed_classes,
            data_dir=self.data_dir,
            n_classes_sup=self.n_classes_sup
        )
        
        test_loader = DataLoader(
            test_dataset,
            shuffle=False,
            batch_size=batch_size,
            num_workers=n_worker,
        )
        eval_dict_sup, eval_dict_sub, eval_dict_sup_only = self.evaluation(test_loader, self.criterion, end_task)
        
        
        print('super class')
        self.report_test(sample_num, eval_dict_sup["avg_loss"], eval_dict_sup["avg_acc"])
        print('super class only')
        self.report_test(sample_num, eval_dict_sup_only["avg_loss"], eval_dict_sup_only["avg_acc"])        
        print('sub class')
        self.report_test(sample_num, eval_dict_sub["avg_loss"], eval_dict_sub["avg_acc"])
        
        return eval_dict_sup, eval_dict_sub, eval_dict_sup_only

    def online_before_task(self, cur_iter):
        # Task-Free
        pass

    def online_after_task(self, cur_iter):
        # Task-Free
        pass

    def reservoir_memory(self, sample):
        self.seen += 1
        if len(self.memory.images) >= self.memory_size:
            j = np.random.randint(0, self.seen)
            if j < self.memory_size:
                self.memory.replace_sample(sample, j)
        else:
            self.memory.replace_sample(sample)

    def reset_opt(self):
        self.optimizer = select_optimizer(self.opt_name, self.lr, self.model)
        self.scheduler = select_scheduler(self.sched_name, self.optimizer, self.lr_gamma)

    def evaluation(self, test_loader, criterion, end_task):
        total_correct_sup, total_num_data_sup, total_loss_sup = 0.0, 0.0, 0.0
        correct_l_sup = torch.zeros(self.n_classes)
        num_data_l_sup = torch.zeros(self.n_classes)
        label_sup = []
        
        
        """performance on superclass only """
        total_correct_sup_only, total_num_data_sup_only, total_loss_sup_only = 0.0, 0.0, 0.0
        correct_l_sup_only = torch.zeros(self.n_classes)
        num_data_l_sup_only = torch.zeros(self.n_classes)
        label_sup_only = []
        """                               """        
        
        
        total_correct_sub, total_num_data_sub, total_loss_sub = 0.0, 0.0, 0.0
        correct_l_sub = torch.zeros(self.n_classes)
        num_data_l_sub = torch.zeros(self.n_classes)
        label_sub = []
        
        self.corresponding_super = torch.tensor(self.corresponding_super).to(self.device)            
        self.y = []
        self.pred_sup = []
        self.pred_sub = []
        self.hierarhcy = []
        
        
        
        self.model.eval()
        with torch.no_grad():
            for i, data in enumerate(test_loader):
                x = data["image"]
                y = data["label"]
                hierarchy = data["hierarchy"]
                cls_sup = data["cls_sup"]
                cls_sub = data["cls_sub"]
                
    
                ##### y: 0~9 for super
                ##### y: 0~99 for sub
                
                x = x.to(self.device)
                y = y.to(self.device)
                hierarchy = hierarchy.to(self.device)
                
                
                logit_sup, logit_sub = self.model(x)
                
                idx_sup = (hierarchy == 0)
                idx_sub = (hierarchy == 1)


                loss_sup = self.criterion(logit_sup[idx_sup], y[idx_sup])
                loss_sub = self.criterion(logit_sub[idx_sub], y[idx_sub])
                
                loss = (loss_sup + loss_sub)/x.size(0)
                
                pred_sup = torch.argmax(logit_sup, dim=-1)
                _, preds_sup = logit_sup.topk(self.topk, 1, True, True)
                
                pred_sub = torch.argmax(logit_sub, dim=-1)
                _, preds_sub = logit_sub.topk(self.topk, 1, True, True)
                
                self.y.append(y)
                self.pred_sup.append(pred_sup)
                self.pred_sub.append(pred_sub)
                
                
                """ for super class """
                if self.model.fc_sub.weight.shape[0] > 1:
                    correct_from_sup = pred_sup[idx_sup] == y[idx_sup]
                    
                    pred_by_subclass =  self.corresponding_super[pred_sub[idx_sup]]
                    
                    correct_from_sub = pred_by_subclass == y[idx_sup]
                    
                    total_correct_sup += torch.sum(torch.logical_or(correct_from_sup, correct_from_sub).double()).item()
                    xlabel_cnt_sup, correct_xlabel_cnt_sup = self._interpret_pred(y[idx_sup], pred_sup[idx_sup], pred_sub[idx_sup])
                    
                else:
                    total_correct_sup += torch.sum(preds_sup[idx_sup] == y[idx_sup].unsqueeze(1)).item()
                    xlabel_cnt_sup, correct_xlabel_cnt_sup = self._interpret_pred(y[idx_sup], pred_sup[idx_sup])
                
                total_num_data_sup += y[idx_sup].size(0)
                correct_l_sup += correct_xlabel_cnt_sup.detach().cpu()
                num_data_l_sup += xlabel_cnt_sup.detach().cpu()
                total_loss_sup += loss_sup.item()
                label_sup += y[idx_sup].tolist()
                
                
                """ for sub class """
                total_correct_sub += torch.sum(preds_sub[idx_sub] == y[idx_sub].unsqueeze(1)).item()
                xlabel_cnt_sub, correct_xlabel_cnt_sub = self._interpret_pred(y[idx_sub], pred_sub[idx_sub])
               
                total_num_data_sub += y[idx_sub].size(0)
                correct_l_sub += correct_xlabel_cnt_sub.detach().cpu()
                num_data_l_sub += xlabel_cnt_sub.detach().cpu()
                total_loss_sub += loss_sub.item()
                label_sub += y[idx_sub].tolist()

                
                """ for super class only """
                total_correct_sup_only += torch.sum(preds_sup[idx_sup] == y[idx_sup].unsqueeze(1)).item()
                xlabel_cnt_sup_only, correct_xlabel_cnt_sup_only = self._interpret_pred(y[idx_sup], pred_sup[idx_sup])

                total_num_data_sup_only += y[idx_sup].size(0)
                correct_l_sup_only += correct_xlabel_cnt_sup_only.detach().cpu()
                num_data_l_sup_only += xlabel_cnt_sup_only.detach().cpu()
                total_loss_sup_only += loss_sup.item()
                label_sup_only += y[idx_sup].tolist()
                                
        self.corresponding_super = list(self.corresponding_super.to('cpu').numpy())            
        self.y = torch.cat(self.y)
        self.pred_sup = torch.cat(self.pred_sup)
        self.pred_sub = torch.cat(self.pred_sub)
        

        ret_sup = self.get_avg_res(total_num_data_sup, total_loss_sup, total_correct_sup, correct_l_sup, num_data_l_sup)        
        ret_sub = self.get_avg_res(total_num_data_sub, total_loss_sub, total_correct_sub, correct_l_sub, num_data_l_sub)
        ret_sup_only = self.get_avg_res(total_num_data_sup_only, total_loss_sup_only, total_correct_sup_only, correct_l_sup_only, num_data_l_sup_only)

        print('save result for task'+str(self.cur_iter+1))
        self.save_results(ret_sup, ret_sub, ret_sup_only, end_task)
        self.save_results(ret_sup, ret_sub, ret_sup_only, end_task, islatest=True)
        
            
        return ret_sup, ret_sub, ret_sup_only        
     
    
    def get_avg_res(self, total_num_data, total_loss, total_correct, correct_l, num_data_l):
        total_loss = 0 if total_num_data == 0 else total_loss
        total_num_data = total_num_data if total_num_data > 0 else float('inf')
        
        avg_acc = total_correct / total_num_data
        avg_loss = total_loss / total_num_data
        cls_acc = (correct_l / (num_data_l + 1e-5)).numpy().tolist()

        if self.check_stream == 1:
            ret = {"total_num_data": total_num_data, "total_loss": total_loss, "total_correct": total_correct, "correct_l": correct_l.numpy(), "num_data_l": num_data_l.numpy(),
                    "avg_loss": avg_loss, "avg_acc": avg_acc, "cls_acc": cls_acc,
                   "exposed_classes": self.exposed_classes, "exposed_classes_sup": self.exposed_classes_sup, "exposed_classes_sub": self.exposed_classes_sub, "corresponding_super": self.corresponding_super,
                    "y":self.y.to('cpu').numpy(), "pred_sup":self.pred_sup.to('cpu').numpy(), "pred_sub":self.pred_sub.to('cpu').numpy()}
        
        else:
            ret = {"total_num_data": total_num_data, "total_loss": total_loss, "total_correct": total_correct, "correct_l": correct_l.numpy(), "num_data_l": num_data_l.numpy(),
                    "avg_loss": avg_loss, "avg_acc": avg_acc, "cls_acc": cls_acc,
                   "exposed_classes": self.exposed_classes, "exposed_classes_sup": self.exposed_classes_sup, "exposed_classes_sub": self.exposed_classes_sub, "corresponding_super": self.corresponding_super,}
                    
            
            
        return ret


    def _interpret_pred(self, y, pred, pred2=None):
        # xlable is batch
        ret_num_data = torch.zeros(self.n_classes)
        ret_corrects = torch.zeros(self.n_classes)

        xlabel_cls, xlabel_cnt = y.unique(return_counts=True)
        for cls_idx, cnt in zip(xlabel_cls, xlabel_cnt):
            ret_num_data[cls_idx] = cnt

        if pred2 != None:
            correct_from_1 = (y == pred)
            
            pred_by_subclass = self.corresponding_super[pred2]
            correct_from_2 = (y == pred_by_subclass)
            mask = torch.logical_or(correct_from_1, correct_from_2)
        
        else:
            mask = (y==pred)
           
        correct_xlabel = y.masked_select(mask)
        correct_cls, correct_cnt = correct_xlabel.unique(return_counts=True)
        for cls_idx, cnt in zip(correct_cls, correct_cnt):
            ret_corrects[cls_idx] = cnt

        return ret_num_data, ret_corrects

    
    def save_results(self, ret_sup, ret_sub, ret_sup_only, end_task, islatest=False):
        
        if islatest:
            folder_name = os.path.join(f"{self.root}/{self.exp_name}/results", self.dataset, self.mode, f'memory_size_{self.memory_size_total}', f'seed_{self.seed}', 'latest')
        else:
            folder_name = os.path.join(f"{self.root}/{self.exp_name}/results", self.dataset, self.mode, f'memory_size_{self.memory_size_total}', f'seed_{self.seed}', self.tm)

        os.makedirs(folder_name, exist_ok=True)
        
        str_ = 'res_task_end_' if end_task else 'res_task_'        
        
        fn_sup = os.path.join(folder_name, str_+str(self.cur_iter)+'_'+str(self.n_count_num)+'_sup.pt')
        torch.save(ret_sup, fn_sup)

        fn_sub = os.path.join(folder_name, str_+str(self.cur_iter)+'_'+str(self.n_count_num)+'_sub.pt')
        torch.save(ret_sub, fn_sub)

        fn_sup_only = os.path.join(folder_name, str_+str(self.cur_iter)+'_'+str(self.n_count_num)+'_sup_only.pt')
        torch.save(ret_sup_only, fn_sup_only)            
    
        if end_task:
            fn_ckpt = os.path.join(folder_name, 'model_task_'+str(self.cur_iter)+'.pt')
            torch.save(self.model.state_dict(), fn_ckpt)
            
