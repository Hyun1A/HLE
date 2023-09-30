import logging
import random
import copy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from scipy.stats import ttest_ind

from methods.er_baseline import ER
from utils.data_loader import cutmix_data, ImageDataset, StreamDataset, MemoryDataset

logger = logging.getLogger()
writer = SummaryWriter("tensorboard")

class CLIB(ER):
    def __init__(
        self, criterion, device, train_transform, test_transform, n_classes, n_classes_sup, n_classes_sub, writer, **kwargs
    ):
        super().__init__(
            criterion, device, train_transform, test_transform, n_classes, n_classes_sup, n_classes_sub, writer, **kwargs
        )

        

        # Samplewise importance variables
        self.loss = np.array([])
        self.dropped_idx = []
        self.memory_dropped_idx = []
        self.imp_update_counter = 0
        self.memory = MemoryDataset(self.root, self.dataset, self.train_transform, self.exposed_classes,
                                    test_transform=self.test_transform, data_dir=self.data_dir, device=self.device,
                                    transform_on_gpu=self.gpu_transform, n_classes_sup=self.n_classes_sup, save_test='cpu', keep_history=True)
        self.imp_update_period = self.online_iter * self.temp_batchsize#kwargs['imp_update_period']
        if kwargs["sched_name"] == 'default':
            self.sched_name = 'adaptive_lr'

        # Adaptive LR variables
        self.lr_step = kwargs["lr_step"]
        self.lr_length = kwargs["lr_length"]
        self.lr_period = kwargs["lr_period"]
        self.prev_loss = None
        self.lr_is_high = True
        self.high_lr = self.lr
        self.low_lr = self.lr_step * self.lr
        self.high_lr_loss = []
        self.low_lr_loss = []
        self.current_lr = self.lr
        
        self.before_subcls = True

    def online_step(self, sample, sample_num, n_worker):
        self.n_count_num +=1
        
        if sample['klass'] not in self.exposed_classes:
            self.add_new_class(sample['klass'], sample['hierarchy'], sample['klass_sup'])
            
        self.num_updates += self.online_iter
        self.temp_batch.append(sample)
        
        if len(self.temp_batch) == self.temp_batchsize:
            for stored_sample in self.temp_batch:
                self.update_memory(stored_sample)  
                
            train_loss_sup, train_acc_sup, train_loss_sub, train_acc_sub = \
                                                      self.online_train([], self.batch_size, n_worker, \
                                                      iterations=int(self.num_updates), stream_batch_size=0)            


            print('super class')
            self.report_training(sample_num, train_loss_sup, train_acc_sup)
            print('sub class')
            self.report_training(sample_num, train_loss_sub, train_acc_sub)                    
                        
                
          
            self.temp_batch = []
            self.num_updates -= int(self.num_updates)
            self.update_schedule()

    def update_memory(self, sample):
        if np.random.rand(1).item() < 0.5:
            self.samplewise_importance_memory(sample)

    
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

            self.samplewise_loss_update()
            
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
                    
            """"""
            sdict = copy.deepcopy(self.optimizer.state_dict())
            fc_params = sdict['param_groups'][self.sup_param_idx+1]['params']
            if len(sdict['state']) > 0:
                if self.before_subcls == True:
                    self.len_state_before_subcls = len(sdict['state'])
                    self.before_subcls = False
                
                fc_weight_state = sdict['state'][fc_params[0]]
                fc_bias_state = sdict['state'][fc_params[1]]
              
            """"""
                
            for param in self.optimizer.param_groups[self.sup_param_idx+1]['params']:
                if param in self.optimizer.state.keys():
                    del self.optimizer.state[param]
            del self.optimizer.param_groups[self.sup_param_idx+1]
            self.optimizer.add_param_group({'params': self.model.fc_sup.parameters()})
            
            if self.sup_param_idx == 0:
                self.sup_param_idx = (self.sup_param_idx+1)%2
                self.sub_param_idx = (self.sub_param_idx+1)%2                  
            
            """"""
            if len(sdict['state']) > 0:
                if 'adam' in self.opt_name:
                    
                    fc_weight = self.optimizer.param_groups[self.sup_param_idx+1]['params'][0]
                    fc_bias = self.optimizer.param_groups[self.sup_param_idx+1]['params'][1]
                    self.optimizer.state[fc_weight]['step'] = fc_weight_state['step']
                    self.optimizer.state[fc_weight]['exp_avg'] = torch.cat([fc_weight_state['exp_avg'],
                                                                            torch.zeros([1, fc_weight_state['exp_avg'].size(
                                                                                dim=1)]).to(self.device)], dim=0)
                    self.optimizer.state[fc_weight]['exp_avg_sq'] = torch.cat([fc_weight_state['exp_avg_sq'],
                                                                               torch.zeros([1, fc_weight_state[
                                                                                   'exp_avg_sq'].size(dim=1)]).to(
                                                                                   self.device)], dim=0)
                    self.optimizer.state[fc_bias]['step'] = fc_bias_state['step']
                    self.optimizer.state[fc_bias]['exp_avg'] = torch.cat([fc_bias_state['exp_avg'],
                                                                          torch.tensor([0]).to(
                                                                              self.device)], dim=0)
                    self.optimizer.state[fc_bias]['exp_avg_sq'] = torch.cat([fc_bias_state['exp_avg_sq'],
                                                                             torch.tensor([0]).to(
                                                                                 self.device)], dim=0)
            """"""

            self.memory.add_new_class(cls_list=self.exposed_classes)  
                
        else:
            self.num_learned_class_sub += 1
            self.exposed_classes_sub.append(class_name)  
            self.corresponding_super.append(self.exposed_classes.index(class_name_sup))
            
            prev_weight = copy.deepcopy(self.model.fc_sub.weight.data)
            self.model.fc_sub = nn.Linear(self.model.fc_sub.in_features, self.num_learned_class_sub).to(self.device)

            with torch.no_grad():
                if self.num_learned_class_sub > 1:
                    self.model.fc_sub.weight[:self.num_learned_class_sub - 1] = prev_weight
                    
            """"""
            sdict = copy.deepcopy(self.optimizer.state_dict())
            fc_params = sdict['param_groups'][self.sub_param_idx+1]['params']
            try:
                fc_weight_state = sdict['state'][fc_params[0]]
                fc_bias_state = sdict['state'][fc_params[1]]
            except:
                pass
            
            """"""
                
            for param in self.optimizer.param_groups[self.sub_param_idx+1]['params']:
                if param in self.optimizer.state.keys():
                    del self.optimizer.state[param]
            del self.optimizer.param_groups[self.sub_param_idx+1]
            self.optimizer.add_param_group({'params': self.model.fc_sub.parameters()})
            
            if self.sub_param_idx == 0:
                self.sup_param_idx = (self.sup_param_idx+1)%2
                self.sub_param_idx = (self.sub_param_idx+1)%2                  
            
            """"""
            try:
                if 'adam' in self.opt_name:

                    fc_weight = self.optimizer.param_groups[self.sub_param_idx+1]['params'][0]
                    fc_bias = self.optimizer.param_groups[self.sub_param_idx+1]['params'][1]
                    self.optimizer.state[fc_weight]['step'] = fc_weight_state['step']
                    self.optimizer.state[fc_weight]['exp_avg'] = torch.cat([fc_weight_state['exp_avg'],
                                                                            torch.zeros([1, fc_weight_state['exp_avg'].size(
                                                                                dim=1)]).to(self.device)], dim=0)
                    self.optimizer.state[fc_weight]['exp_avg_sq'] = torch.cat([fc_weight_state['exp_avg_sq'],
                                                                               torch.zeros([1, fc_weight_state[
                                                                                   'exp_avg_sq'].size(dim=1)]).to(
                                                                                   self.device)], dim=0)
                    self.optimizer.state[fc_bias]['step'] = fc_bias_state['step']
                    self.optimizer.state[fc_bias]['exp_avg'] = torch.cat([fc_bias_state['exp_avg'],
                                                                          torch.tensor([0]).to(
                                                                              self.device)], dim=0)
                    self.optimizer.state[fc_bias]['exp_avg_sq'] = torch.cat([fc_bias_state['exp_avg_sq'],
                                                                             torch.tensor([0]).to(
                                                                                 self.device)], dim=0)
            except:
                pass
            """"""

            self.memory.add_new_class(cls_list=self.exposed_classes)  
                        
        if 'reset' in self.sched_name:
            self.update_schedule(reset=True)

            
    def update_schedule(self, reset=False):
        if self.sched_name == 'adaptive_lr':
            self.adaptive_lr(period=self.lr_period, min_iter=self.lr_length)
            self.model.train()
        else:
            super().update_schedule(reset)

    def samplewise_loss_update(self, ema_ratio=0.99, batchsize=512):
        self.imp_update_counter += 1
        if self.imp_update_counter % self.imp_update_period == 0:
            #print(self.imp_update_counter)
            if len(self.memory) > 0:
                self.model.eval()
                with torch.no_grad():
                    x = self.memory.device_img
                    y = torch.LongTensor(self.memory.labels)
                    y = y.to(self.device)
                    hierarchy = torch.LongTensor(self.memory.hierarchy)
                    
                    if self.use_amp:
                        with torch.cuda.amp.autocast():
                            
                            res = [self.model(torch.cat(x[i * batchsize:min((i + 1) * batchsize, len(x))]).to(self.device)) \
                                for i in range(-(-len(x) // batchsize))]
                            
                            if len(res) > 1:

                                logit_sup = [res[i][0] for i in range(len(res))]
                                logit_sub = [res[i][1] for i in range(len(res))]
                                
                                logit_sup = torch.cat(logit_sup, dim=0)
                                logit_sub = torch.cat(logit_sub, dim=0)   
                                
                            else:
                                logit_sup, logit_sub = res[0]

                    else:
                        logit_sup, logit_sub = \
                            zip([self.model(torch.cat(x[i * batchsize:min((i + 1) * batchsize, len(x))]).to(self.device)) \
                            for i in range(-(-len(x) // batchsize))])

                        logit_sup = torch.cat(logit_sup, dim = 0)
                        logit_sub = torch.cat(logit_sub, dim = 0)

                    idx_sup = (hierarchy == 0)
                    idx_sub = (hierarchy == 1)
                    loss = torch.zeros_like(hierarchy).cpu().numpy()
                        
                    loss_sup = F.cross_entropy(logit_sup[idx_sup], y[idx_sup], reduction='none').cpu().numpy() \
                        if torch.sum(idx_sup.type(torch.float)) != 0 else np.array([0.])                    
                    
                    loss_sub = F.cross_entropy(logit_sub[idx_sub], y[idx_sub]-self.memory.n_classes_sup, reduction='none').cpu().numpy() \
                        if torch.sum(idx_sub.type(torch.float)) != 0 else np.array([0.])                        
                    loss[idx_sup] = loss_sup
                    loss[idx_sub] = loss_sub
                    
                    
                self.memory.update_loss_history(loss, self.loss, ema_ratio=ema_ratio, dropped_idx=self.memory_dropped_idx)
                self.memory_dropped_idx = []
                self.loss = loss

    def samplewise_importance_memory(self, sample):
        if len(self.memory.images) >= self.memory_size:
            label_frequency = copy.deepcopy(self.memory.cls_count)
            label_frequency[self.exposed_classes.index(sample['klass'])] += 1
            cls_to_replace = np.argmax(np.array(label_frequency))
            cand_idx = self.memory.cls_idx[cls_to_replace]
            score = self.memory.others_loss_decrease[cand_idx]
            idx_to_replace = cand_idx[np.argmin(score)]
            self.memory.replace_sample(sample, idx_to_replace)
            self.dropped_idx.append(idx_to_replace)
            self.memory_dropped_idx.append(idx_to_replace)
        else:
            self.memory.replace_sample(sample)
            self.dropped_idx.append(len(self.memory) - 1)
            self.memory_dropped_idx.append(len(self.memory) - 1)

    def adaptive_lr(self, period=10, min_iter=10, significance=0.05):
        if self.imp_update_counter % self.imp_update_period == 0:
            self.train_count += 1
            mask = np.ones(len(self.loss), bool)
            mask[self.dropped_idx] = False
            if self.train_count % period == 0:
                if self.lr_is_high:
                    if self.prev_loss is not None and self.train_count > 20:
                        self.high_lr_loss.append(np.mean((self.prev_loss - self.loss[:len(self.prev_loss)])[mask[:len(self.prev_loss)]]))
                        if len(self.high_lr_loss) > min_iter:
                            del self.high_lr_loss[0]
                    self.prev_loss = self.loss
                    self.lr_is_high = False
                    for param_group in self.optimizer.param_groups:
                        param_group["lr"] = self.low_lr
                        param_group["initial_lr"] = self.low_lr
                else:
                    if self.prev_loss is not None and self.train_count > 20:
                        self.low_lr_loss.append(np.mean((self.prev_loss - self.loss[:len(self.prev_loss)])[mask[:len(self.prev_loss)]]))
                        if len(self.low_lr_loss) > min_iter:
                            del self.low_lr_loss[0]
                    self.prev_loss = self.loss
                    self.lr_is_high = True
                    for param_group in self.optimizer.param_groups:
                        param_group["lr"] = self.high_lr
                        param_group["initial_lr"] = self.high_lr
                self.dropped_idx = []
                if len(self.high_lr_loss) == len(self.low_lr_loss) and len(self.high_lr_loss) >= min_iter:
                    stat, pvalue = ttest_ind(self.low_lr_loss, self.high_lr_loss, equal_var=False, alternative='greater')
                    print(pvalue)
                    if pvalue < significance:
                        self.high_lr = self.low_lr
                        self.low_lr *= self.lr_step
                        self.high_lr_loss = []
                        self.low_lr_loss = []
                        if self.lr_is_high:
                            self.lr_is_high = False
                            for param_group in self.optimizer.param_groups:
                                param_group["lr"] = self.low_lr
                                param_group["initial_lr"] = self.low_lr
                        else:
                            self.lr_is_high = True
                            for param_group in self.optimizer.param_groups:
                                param_group["lr"] = self.high_lr
                                param_group["initial_lr"] = self.high_lr
                    elif pvalue > 1 - significance:
                        self.low_lr = self.high_lr
                        self.high_lr /= self.lr_step
                        self.high_lr_loss = []
                        self.low_lr_loss = []
                        if self.lr_is_high:
                            self.lr_is_high = False
                            for param_group in self.optimizer.param_groups:
                                param_group["lr"] = self.low_lr
                                param_group["initial_lr"] = self.low_lr
                        else:
                            self.lr_is_high = True
                            for param_group in self.optimizer.param_groups:
                                param_group["lr"] = self.high_lr
                                param_group["initial_lr"] = self.high_lr