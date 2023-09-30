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
        self.depth = kwargs['depth']
        
        self.num_learned_class_hier = torch.zeros(self.depth+1)

        self.param_idx = np.arange(self.depth+1)

        self.num_learning_class = 1
        self.n_classes = n_classes
        self.n_classes_sup = n_classes_sup
        self.n_classes_sub = n_classes_sub
        
        self.exposed_classes = []
        self.check_stream = 0
        self.exposed_classes_sup = []
        self.exposed_classes_sub = []
        
        self.exposed_classes_hier = [[] for i in range(self.depth+1)]
        self.exposed_hierarchies = []
        
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
            
        self.model = select_model(self.model_name, self.dataset, num_heads=self.depth+1).to(self.device)
        self.optimizer = select_optimizer(self.opt_name, self.lr, self.model)
        
        if 'imagenet' in self.dataset:
            self.lr_gamma = 0.9999
        else:
            self.lr_gamma = 0.9999
        self.scheduler = select_scheduler(self.sched_name, self.optimizer, self.lr_gamma)

        self.criterion = criterion.to(self.device)
        self.memory = MemoryDataset(self.root, self.dataset, self.train_transform, self.exposed_classes,
                                    test_transform=self.test_transform, data_dir=self.data_dir, device=self.device,
                                    transform_on_gpu=self.gpu_transform, n_classes_sup=self.n_classes_sup, 
                                    hierarchy_list=self.exposed_hierarchies, depth=self.depth)
                
        self.temp_batch = []
        self.num_updates = 0
        self.train_count = 0
        self.batch_size = kwargs["batchsize"]

        self.start_time = time.time()
        num_samples = {'cifar10': 50000, 'cifar100': 50000, 'tinyimagenet': 100000, 'imagenet': 1281167, 'imagenet_subset': 128741*2, 'imagenet_subset_sub_shuffle': 128741*2, 'cifar100_hier_setup1':100000,  'cifar100_hier_setup2':50000, 'cifar100_hier_setup3':50000, 'stanford_car_setup1':8144*2, 'imagenet_subset_setup2': 128741, 'stanford_car_setup2':8144, 'cub_200_2011_scenario1': 5994, "cifar100_scene_topdown": 50000, "cifar100_scene_bottomup": 50000, "inat19_scene_topdown": 187385}
        self.total_samples = num_samples[self.dataset]
        
    def online_step(self, sample, sample_num, n_worker):
        self.n_count_num +=1
        
        if sample['klass'] not in self.exposed_classes:
            hierarchy = sample['hierarchy']
            self.add_new_class(sample['klass'], hierarchy, sample[f'klass_{hierarchy}'])

        self.temp_batch.append(sample)
        self.num_updates += self.online_iter

        if len(self.temp_batch) == self.temp_batchsize:
            if self.check_stream == 0:
                train_loss_hier, train_acc_hier = self.online_train(self.temp_batch, self.batch_size, n_worker, \
                                                      iterations=int(self.num_updates), stream_batch_size=self.temp_batchsize)                
            
            for h in range(self.depth+1):
                print(f'hierarchy {h}')
                self.report_training(sample_num, train_loss_hier[h], train_acc_hier[h])

            for stored_sample in self.temp_batch:
                self.update_memory(stored_sample)
            self.temp_batch = []
            self.num_updates -= int(self.num_updates)
            
            
    def add_new_class(self, class_name, hierarchy, class_name_sup=None, class_name_sub=None):
        self.exposed_classes.append(class_name)
        self.exposed_hierarchies.append(hierarchy)
        self.num_learned_class = len(self.exposed_classes)

        self.num_learned_class_hier[hierarchy] += 1
        self.exposed_classes_hier[hierarchy].append(class_name)
        
        prev_weight = copy.deepcopy(self.model.fc[hierarchy].weight.data)
        self.model.fc[hierarchy] =  nn.Linear(self.model.fc[hierarchy].in_features, int(self.num_learned_class_hier[hierarchy].item())  ).to(self.device) 
        
        with torch.no_grad():
            if self.num_learned_class_hier[hierarchy] > 1:
                self.model.fc[hierarchy].weight[:int(self.num_learned_class_hier[hierarchy].item()) - 1] = prev_weight

        for param in self.optimizer.param_groups[self.param_idx[hierarchy]+1]['params']:
            if param in self.optimizer.state.keys():
                del self.optimizer.state[param]
        del self.optimizer.param_groups[self.param_idx[hierarchy]+1]
        self.optimizer.add_param_group({'params': self.model.fc[hierarchy].parameters()})
        self.memory.add_new_class(cls_list=self.exposed_classes, hierarchy_list=self.exposed_hierarchies)

        for h in range(self.depth+1):
            if (h != hierarchy) and (self.param_idx[h] > self.param_idx[hierarchy]):
                self.param_idx[h] -= 1
        
        self.param_idx[hierarchy] = self.depth    

        if 'reset' in self.sched_name:
            self.update_schedule(reset=True)
            
    def online_train(self, sample, batch_size, n_worker, iterations=1, stream_batch_size=1):

        total_loss_hier, correct_hier, num_data_hier = np.zeros(self.depth+1), np.zeros(self.depth+1), np.zeros(self.depth+1)

        if stream_batch_size > 0:
            sample_dataset = StreamDataset(self.root, sample, dataset=self.dataset, transform=self.train_transform, \
                                           cls_list=self.exposed_classes, data_dir=self.data_dir, device=self.device, \
                                           transform_on_gpu=self.gpu_transform, n_classes_sup=self.n_classes_sup,\
                                           hierarchy_list=self.exposed_hierarchies, depth=self.depth)
            
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

            logit_hier, loss_hier = self.model_forward(x,y, hierarchy)

            loss = torch.tensor([0.]).to(self.device)
            for l in loss_hier:
                loss+=l
            loss /= x.size(0)


            if self.use_amp:
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                self.optimizer.step()

            self.update_schedule()

            preds_hier = []
            for logit in logit_hier:
                _, preds = logit.topk(self.topk, 1, True, True)
                preds_hier.append(preds)
            
            idx_hier = []
            for h in range(self.depth+1):
                idx_hier.append((hierarchy==h))
                                           
            for h in range(self.depth+1):
                total_loss_hier[h] += loss_hier[h].item()
                correct_hier[h] += torch.sum(preds_hier[h][idx_hier[h]] == y[idx_hier[h]].unsqueeze(1)).item()
                num_data_hier[h] += y[idx_hier[h]].size(0)           

        train_loss_hier = []
        train_acc_hier = []
        for h in range(self.depth+1):
            train_loss_hier.append( total_loss_hier[h] / num_data_hier[h] if num_data_hier[h] != 0 else 0. )
            train_acc_hier.append( correct_hier[h] / num_data_hier[h] if num_data_hier[h] != 0 else 0. )
            
        return train_loss_hier, train_acc_hier
    
    
    def model_forward(self, x, y, hierarchy):
        do_cutmix = self.cutmix and np.random.rand(1) < 0.5

        idx_hier = []
        num_idx_hier = []
        
        for h in range(self.depth+1):
            idx_hier.append((hierarchy==h))
            num_idx_hier.append(  torch.sum(idx_hier[h].type(torch.float))  )

        if do_cutmix:
            if self.use_amp:
                with torch.cuda.amp.autocast():
                    labels_a = deepcopy(y)
                    labels_b = deepcopy(y)

                                
                    x_hier, labels_a_hier, labels_b_hier, lam_hier = [], [], [], []
                    for h in range(self.depth+1):
                        x_h, labels_a_h, labels_b_h, lam_h = cutmix_data(x=x[idx_hier[h]], y=y[idx_hier[h]], alpha=1.0)
                        x_hier.append(x_h)
                        labels_a_hier.append(labels_a_h)
                        labels_b_hier.append(labels_b_h)
                        lam_hier.append(lam_h)
                        
                                        
                    for h in range(self.depth+1):
                        x[idx_hier[h]] = x_hier[h]
                        labels_a[idx_hier[h]] = labels_a_hier[h]
                        labels_b[idx_hier[h]] = labels_b_hier[h]

                    logit_hier = self.model(x)                  
                        
                    loss_hier = []   
                    for h in range(self.depth+1):
                        loss_hier.append( 
                                          lam_hier[h]*self.criterion(logit_hier[h][idx_hier[h]], labels_a[idx_hier[h]]) + \
                                          (1 - lam_hier[h])*self.criterion(logit_hier[h][idx_hier[h]], labels_b[idx_hier[h]]) \
                                          if torch.sum(idx_hier[h].type(torch.float)) != 0 else torch.tensor([0.]).to(self.device)
                                        )

        else:            
            if self.use_amp:
                with torch.cuda.amp.autocast():
                    logit_hier = self.model(x)     

                    loss_hier = []   
                    for h in range(self.depth+1):
                        loss_hier.append( 
                                          self.criterion(logit_hier[h][idx_hier[h]], y[idx_hier[h]]) \
                                          if torch.sum(idx_hier[h].type(torch.float)) != 0 else torch.tensor([0.]).to(self.device)
                                        )

        return logit_hier, loss_hier

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

        test_dataset = ImageDataset(
            self.root,
            exp_test_df,
            dataset=self.dataset,
            transform=self.test_transform,
            cls_list=self.exposed_classes,
            data_dir=self.data_dir,
            n_classes_sup=self.n_classes_sup,
            hierarchy_list=self.exposed_hierarchies,
            depth=self.depth
        )
        test_loader = DataLoader(
            test_dataset,
            shuffle=False,
            batch_size=batch_size,
            num_workers=n_worker,
        )
        
        eval_dict_hier = self.evaluation(test_loader, self.criterion, end_task)
        
        for h in range(self.depth+1):
            print(f'hierarchy {h}')
            self.report_test(sample_num, eval_dict_hier[h]["avg_loss"], eval_dict_hier[h]["avg_acc"])

        return eval_dict_hier

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
        """ performance using fine-grained classes """
        total_correct_hier, total_num_data_hier, total_loss_hier = {}, {}, {}
        correct_l_hier, num_data_l_hier, label_hier = {}, {}, {}
        
        for h in range(self.depth+1):
            total_correct_hier[h], total_num_data_hier[h], total_loss_hier[h] = 0.0, 0.0, 0.0
            correct_l_hier[h], num_data_l_hier[h], label_hier[h] = torch.zeros(self.n_classes), torch.zeros(self.n_classes), []
                
        self.y = []
        self.preds_hier = [[] for i in range(self.depth+1)]
        self.hierarhcy = []

        self.model.eval()
        len_ = 0
        with torch.no_grad():
            for i, data in enumerate(test_loader):
                x = data["image"]
                y = data["label"]
                hierarchy = data["hierarchy"]
                
                len_ += len(y)

                x = x.to(self.device)
                y = y.to(self.device)
                hierarchy = hierarchy.to(self.device)
                
                idx_hier = []
                for h in range(self.depth+1):
                    idx_hier.append((hierarchy==h))
                
                logit_hier = self.model(x)
                

                loss_hier = []   
                for h in range(self.depth+1):
                    #print(self.model.fc)
                    #print(logit_hier[h].shape)
                    #print(y)
                    #print(h)
                    
                    
                    loss_hier.append( 
                      self.criterion(logit_hier[h][idx_hier[h]], y[idx_hier[h]]) \
                      if torch.sum(idx_hier[h].type(torch.float)) != 0 else torch.tensor([0.]).to(self.device)
                    )        
                    
                loss = torch.tensor([0.]).to(self.device)
                for l in loss_hier:
                    loss+=l
                loss /= x.size(0)

                preds_hier = []
                for logit in logit_hier:
                    _, preds = logit.topk(self.topk, 1, True, True)
                    preds_hier.append(preds)
                
                for h in range(self.depth+1):
                    self.preds_hier[h].append(preds_hier[h])
                
                self.y.append(y)

                
                """ for hierarchies """
                for h in range(self.depth+1):
                    total_correct_hier[h] += torch.sum(preds_hier[h][idx_hier[h]] == y[idx_hier[h]].unsqueeze(1)).item()
                    xlabel_cnt_h, correct_xlabel_cnt_h = self._interpret_pred(y[idx_hier[h]], preds_hier[h][idx_hier[h]])

                    total_num_data_hier[h] += y[idx_hier[h]].size(0)
                    correct_l_hier[h] += correct_xlabel_cnt_h.detach().cpu()
                    num_data_l_hier[h] += xlabel_cnt_h.detach().cpu()
                    total_loss_hier[h] += loss_hier[h].item()
                    label_hier[h] += y[idx_hier[h]].tolist()

        print(len_)
        print(self.num_learned_class_hier)
        print(self.model.fc)
        #input()
        self.y = torch.cat(self.y)
        for h in range(self.depth+1):
            self.preds_hier[h] = torch.cat(self.preds_hier[h])
        

        ret_hier = []
        for h in range(self.depth+1):
            avg_res_h = self.get_avg_res(total_num_data_hier[h], total_loss_hier[h], total_correct_hier[h], correct_l_hier[h], num_data_l_hier[h])
            ret_hier.append(avg_res_h)

        print('save result for task'+str(self.cur_iter+1))
        self.save_results(ret_hier, end_task)
        self.save_results(ret_hier, end_task, islatest=True)
            
        return ret_hier      
     
    
    def get_avg_res(self, total_num_data, total_loss, total_correct, correct_l, num_data_l):
        total_loss = 0 if total_num_data == 0 else total_loss
        total_num_data = total_num_data if total_num_data > 0 else float('inf')
        
        avg_acc = total_correct / total_num_data
        avg_loss = total_loss / total_num_data
        cls_acc = (correct_l / (num_data_l + 1e-5)).numpy().tolist()

        if self.check_stream == 1:
            ret = {"total_num_data": total_num_data, "total_loss": total_loss, "total_correct": total_correct, "correct_l": correct_l.numpy(), "num_data_l": num_data_l.numpy(),
                    "avg_loss": avg_loss, "avg_acc": avg_acc, "cls_acc": cls_acc,
                   "exposed_classes": self.exposed_classes, "exposed_hierarchies": self.exposed_hierarchies,
                    "y":self.y.to('cpu').numpy(), "pred_sup":self.pred_sup.to('cpu').numpy(), "pred_sub":self.pred_sub.to('cpu').numpy()}
        
        else:
            ret = {"total_num_data": total_num_data, "total_loss": total_loss, "total_correct": total_correct, "correct_l": correct_l.numpy(), "num_data_l": num_data_l.numpy(),
                    "avg_loss": avg_loss, "avg_acc": avg_acc, "cls_acc": cls_acc,
                   "exposed_classes": self.exposed_classes, "exposed_hierarchies": self.exposed_hierarchies}
                    

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

    
    def save_results(self, ret_hier, end_task, islatest=False):
        
        if islatest:
            folder_name = os.path.join(f"{self.root}/{self.exp_name}/results", self.dataset, self.mode, f'memory_size_{self.memory_size_total}', f'seed_{self.seed}', 'latest')
        else:
            folder_name = os.path.join(f"{self.root}/{self.exp_name}/results", self.dataset, self.mode, f'memory_size_{self.memory_size_total}', f'seed_{self.seed}', self.tm)

        os.makedirs(folder_name, exist_ok=True)
        
        str_ = 'res_task_end_' if end_task else 'res_task_'        
        
        fn_hier = []
        for h in range(self.depth+1):
            fn = os.path.join(folder_name, str_+str(self.cur_iter)+'_'+str(self.n_count_num)+f'_hier_{h}.pt')
            torch.save(ret_hier[h], fn)            
    
        if end_task:
            fn_ckpt = os.path.join(folder_name, 'model_task_'+str(self.cur_iter)+'.pt')
            torch.save(self.model.state_dict(), fn_ckpt)
            
