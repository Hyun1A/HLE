# @inproceedings{wu2019large,
#   title={Large scale incremental learning},
#   author={Wu, Yue and Chen, Yinpeng and Wang, Lijuan and Ye, Yuancheng and Liu, Zicheng and Guo, Yandong and Fu, Yun},
#   booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
#   pages={374--382},
#   year={2019}
# }
import logging
import copy
from copy import deepcopy
import os

import pandas as pd
import torch
import numpy as np
from torch import nn
from torch.utils.data import DataLoader

from methods.er_baseline import ER
from utils.data_loader import ImageDataset, cutmix_data, StreamDataset
from utils.train_utils import select_model, cycle
from torch.utils.tensorboard import SummaryWriter

logger = logging.getLogger()
#writer = SummaryWriter("tensorboard")


class BiasCorrectionLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(1, 1, bias=True)
        self.linear.weight.data.fill_(1.0)
        self.linear.bias.data.fill_(0.0)

    def forward(self, x):
        #print("In BiasCorrectionLayer")
        #print(x.shape)
        #print('x:', x)
        correction = self.linear(x.unsqueeze(dim=2))
        #print(correction.shape)
        correction = correction.squeeze(dim=2)
        #print(correction.shape)
        
        return correction


class BiasCorrection(ER):
    def __init__(
        self, criterion, device, train_transform, test_transform, n_classes, n_classes_sup, n_classes_sub, writer, **kwargs
    ):
        super().__init__(
            criterion, device, train_transform, test_transform, n_classes, n_classes_sup, n_classes_sub, writer, **kwargs
        )
        """
        self.valid_list: valid set which is used for training bias correction layer.
        self.memory_list: training set only including old classes. As already mentioned in the paper,
            memory list and valid list are exclusive.
        self.bias_layer_list - the list of bias correction layers. The index of the list means the task number.
        """
        
        self.prev_model = select_model(
            self.model_name, self.dataset, 1
        )
        self.bias_layer = None
        self.valid_list = []
        
        if 'stanford_car' in self.dataset:
            self.valid_size = round(self.memory_size * 0.21)
        elif 'cifar100' in self.dataset:
            self.valid_size = round(self.memory_size * 0.1)
        elif 'imagenet_subset' in self.dataset:
            self.valid_size = round(self.memory_size * 0.1)           
        
        self.memory_size = self.memory_size - self.valid_size

        self.n_tasks = kwargs["n_tasks"]
        self.bias_layer_list = []
        for _ in range(self.n_tasks):
            bias_layer = BiasCorrectionLayer().to(self.device)
            self.bias_layer_list.append(bias_layer)
        self.distilling = kwargs["distilling"]

        self.val_per_cls = self.valid_size
        self.val_full = False
        self.cur_iter = 0
        self.bias_labels = []
        self.bias_labels_sup = []
        self.bias_labels_sub = []
        

    def online_before_task(self, cur_iter):
        super().online_before_task(cur_iter)
        self.cur_iter = cur_iter
        self.bias_labels.append([])
        self.bias_labels_sup.append([])
        self.bias_labels_sub.append([])

        
    def online_after_task(self, cur_iter):
        if self.distilling:
            self.prev_model = deepcopy(self.model)

    def online_step(self, sample, sample_num, n_worker):
        self.n_count_num +=1
        
        if sample['klass'] not in self.exposed_classes:
            self.add_new_class(sample['klass'], sample['hierarchy'], sample['klass_sup'])

        use_sample = self.online_valid_update(sample)
        
        self.num_updates += self.online_iter

        if use_sample:
            self.temp_batch.append(sample)
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
        if self.distilling:
            self.prev_model = deepcopy(self.model)
            
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
                
            self.bias_labels_sup[self.cur_iter].append(self.num_learned_class_sup - 1)    
                
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
            
            self.bias_labels_sub[self.cur_iter].append(self.num_learned_class_sub - 1)
        
        self.bias_labels[self.cur_iter].append(self.num_learned_class - 1)
        
        
        if self.num_learned_class > 1:
            #print('len num_learned_class', self.num_learned_class)
            #print('before reducing:', len(self.valid_list))
            self.online_reduce_valid(self.num_learned_class)

        if 'reset' in self.sched_name:
            self.update_schedule(reset=True)

    
    def online_reduce_valid(self, num_learned_class):
        #print('online_reduce_vlalid')
        self.val_per_cls = self.valid_size//num_learned_class
        val_df = pd.DataFrame(self.valid_list)
        #print('in online_reduce_valid:', len(self.valid_list))
        #print(val_df['klass'].unique())
        #print()
        
        valid_list = []
        for klass in val_df["klass"].unique():
            class_val = val_df[val_df.klass == klass]
            if len(class_val) > self.val_per_cls:
                new_class_val = class_val.sample(n=self.val_per_cls)
            else:
                new_class_val = class_val
            valid_list += new_class_val.to_dict(orient="records")
        self.valid_list = valid_list
        self.val_full = False

    def online_valid_update(self, sample):
        #print('online_valid_update')
        #print(len(self.valid_list))
        #print()
        
        val_df = pd.DataFrame(self.valid_list, columns=['klass', 'file_name', 'label'])
        if not self.val_full:
            if len(val_df[val_df["klass"] == sample["klass"]]) < self.val_per_cls:
                self.valid_list.append(sample)
                if len(self.valid_list) == self.val_per_cls*self.num_learned_class:
                    self.val_full = True
                use_sample = False
            else:
                use_sample = True
        else:
            use_sample = True
        return use_sample

    def online_train(self, sample, batch_size, n_worker, iterations=1, stream_batch_size=1):
        self.model.train()
        total_loss_sup, distill_loss, classify_loss, correct_sup, num_data_sup = 0.0, 0.0, 0.0, 0.0, 0.0
        total_loss_sub, distill_loss, classify_loss, correct_sub, num_data_sub = 0.0, 0.0, 0.0, 0.0, 0.0


        if stream_batch_size > 0:
            sample_dataset = StreamDataset(self.root, sample, dataset=self.dataset, transform=self.train_transform,
                                           cls_list=self.exposed_classes, data_dir=self.data_dir, device=self.device,
                                           transform_on_gpu=self.gpu_transform, n_classes_sup=self.n_classes_sup)
            
            
            
            
        if len(self.memory) > 0 and batch_size - stream_batch_size > 0:
            memory_batch_size = min(len(self.memory), batch_size - stream_batch_size)
            
            

        for i in range(iterations):
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
            
            do_cutmix = self.cutmix and np.random.rand(1) < 0.5
            
            idx_sup = (hierarchy == 0)
            idx_sub = (hierarchy == 1)  
            
            num_idx_sup = torch.sum(idx_sup.type(torch.float))
            num_idx_sub = torch.sum(idx_sub.type(torch.float))
            
            if (torch.sum(idx_sup.type(torch.float)) == 1) or (torch.sum(idx_sub.type(torch.float)) == 1):
                do_cutmix=False      
                
            #print(do_cutmix)
            
            if do_cutmix:
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
                
                if self.cur_iter != 0:
                    if self.distilling:
                        with torch.no_grad():
                            if self.use_amp:
                                with torch.cuda.amp.autocast():
                                    logit_old_sup, logit_old_sub = self.prev_model(x)
                                    logit_old_sup = self.online_bias_forward(logit_old_sup, self.cur_iter - 1, 0)
                                    logit_old_sub = self.online_bias_forward(logit_old_sub, self.cur_iter - 1, 1)
                                                                        
                self.optimizer.zero_grad()
                
                if self.use_amp:
                    with torch.cuda.amp.autocast():
                        #logit_new = self.model(x)
                        #loss_c = lam * self.criterion(logit_new, labels_a) + (1 - lam) * self.criterion(
                        #logit_new, labels_b)
                        logit_new_sup, logit_new_sub = self.model(x)
                        
                        loss_c_sup = lam*self.criterion(logit_new_sup[idx_sup], labels_a[idx_sup]) + (1 - lam) * self.criterion(logit_new_sup[idx_sup], labels_b[idx_sup]) \
                            if torch.sum(idx_sup.type(torch.float)) != 0 else torch.tensor([0.]).to(self.device)

                        loss_c_sub = lam*self.criterion(logit_new_sub[idx_sub], labels_a[idx_sub]) + (1 - lam) * self.criterion(logit_new_sub[idx_sub], labels_b[idx_sub]) \
                            if torch.sum(idx_sub.type(torch.float)) != 0 else torch.tensor([0.]).to(self.device)                      
                        
                        loss_c = (loss_c_sup+loss_c_sub)                        
                        
                        
                    
            else:
                if self.cur_iter != 0:
                    if self.distilling:
                        with torch.no_grad():
                            if self.use_amp:
                                with torch.cuda.amp.autocast():
                                    logit_old_sup, logit_old_sub = self.prev_model(x)
                                    logit_old_sup = self.online_bias_forward(logit_old_sup, self.cur_iter - 1, 0)
                                    logit_old_sub = self.online_bias_forward(logit_old_sub, self.cur_iter - 1, 1)

                self.optimizer.zero_grad()
                
                if self.use_amp:
                    with torch.cuda.amp.autocast():
                        logit_new_sup, logit_new_sub = self.model(x)
                        
                        idx_sup = (hierarchy == 0)
                        idx_sub = (hierarchy == 1)

                        loss_c_sup = self.criterion(logit_new_sup[idx_sup], y[idx_sup]) \
                            if torch.sum(idx_sup.type(torch.float)) != 0 else torch.tensor([0.]).to(self.device)

                        loss_c_sub = self.criterion(logit_new_sub[idx_sub], y[idx_sub]) \
                            if torch.sum(idx_sub.type(torch.float)) != 0 else torch.tensor([0.]).to(self.device)                      
                        
                        loss_c = (loss_c_sup+loss_c_sub)

                    

            if self.distilling:
                if self.cur_iter == 0:
                    loss_d_sup = torch.tensor(0.0).to(self.device)
                    loss_d_sub = torch.tensor(0.0).to(self.device)
                    loss_d = torch.tensor(0.0).to(self.device)
                else:
                    loss_d_sup = self.distillation_loss(logit_old_sup, logit_new_sup[:, : logit_old_sup.size(1)]).sum()
                    loss_d_sub = self.distillation_loss(logit_old_sub, logit_new_sub[:, : logit_old_sub.size(1)]).sum()
                    loss_d = (loss_d_sup+loss_d_sub)
                    
            else:
                loss_d = torch.tensor(0.0).to(self.device)
                loss_d_sup = torch.tensor(0.0).to(self.device)
                loss_d_sub = torch.tensor(0.0).to(self.device)
                
            #print('loss_c:', loss_c / x.size(0))
            #print('loss_d:', loss_d / x.size(0))
            #print()
            
            loss = (loss_c + loss_d) / x.size(0)
            
            
            if self.use_amp:
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 10)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 10)
                self.optimizer.step()
            self.update_schedule()
            
            _, preds_sup = logit_new_sup.topk(self.topk, 1, True, True)
            _, preds_sub = logit_new_sub.topk(self.topk, 1, True, True)
            
            idx_sup = (hierarchy == 0)
            idx_sub = (hierarchy == 1)              
            

            #print('loss_c_sup:', loss_c_sup)
            #print('loss_d_sup:', loss_d_sup)
                        
            #idx_sup_pos = idx_sup.type(torch.float).nonzero()
            #idx_sub_pos = idx_sub.type(torch.float).nonzero()              
            
            total_loss_sup += (loss_c_sup+loss_d_sup).item()
            correct_sup += torch.sum(preds_sup[idx_sup] == y[idx_sup].unsqueeze(1)).item()
            num_data_sup += y[idx_sup].size(0)
            
            total_loss_sub += (loss_c_sub+loss_d_sub).item()
            correct_sub += torch.sum(preds_sub[idx_sub] == y[idx_sub].unsqueeze(1)).item()

            num_data_sub += y[idx_sub].size(0)                  
        
            
            
        train_loss_sup = total_loss_sup / num_data_sup if num_data_sup != 0 else 0.
        train_acc_sup = correct_sup / num_data_sup if num_data_sup != 0 else 0.
            
        train_loss_sub = total_loss_sub / num_data_sub if num_data_sub != 0 else 0.
        train_acc_sub = correct_sub / num_data_sub if num_data_sub != 0 else 0.   
        
        

        return train_loss_sup, train_acc_sup, train_loss_sub, train_acc_sub
    
    
    
    

    def online_bias_forward(self, input, iter, hierarchy):
        if hierarchy == 0:
            bias_labels = self.bias_labels_sup[iter]
        elif hierarchy == 1:
            bias_labels = self.bias_labels_sub[iter]        
        
        #print('bias_labels', bias_labels)
        bias_layer = self.bias_layer_list[iter]
        #print('bias_layer:', bias_layer)

        #print(input.shape)
        #print(bias_labels)
        #print(input[:, bias_labels])
        #print(bias_layer(input[:, bias_labels]))
        
        if len(bias_labels) > 0:
            input[:, bias_labels] = bias_layer(input[:, bias_labels])
        return input

    def online_evaluate(self, test_list, sample_num, batch_size, n_worker, end_task=False):
        self.online_bias_correction()
        test_df = pd.DataFrame(test_list)
        
        exp_test_df = test_df[test_df['klass'].isin(self.exposed_classes)]
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
        self.bias_layer.eval()

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

                
                """"""
                logit_sup = self.online_bias_forward(logit_sup, self.cur_iter, 0)
                logit_sub = self.online_bias_forward(logit_sub, self.cur_iter, 1)
                      
                """"""       
                    
                    
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
        
        
        print('super class')
        self.report_test(sample_num, ret_sup["avg_loss"], ret_sup["avg_acc"])
        print('super class only')
        self.report_test(sample_num, ret_sup_only["avg_loss"], ret_sup_only["avg_acc"])        
        print('sub class')
        self.report_test(sample_num, ret_sub["avg_loss"], ret_sub["avg_acc"])        
            
        return ret_sup, ret_sub, ret_sup_only        
     
    
    
    
    def online_bias_correction(self, n_iter=256, batch_size=100, n_worker=4):
        self.bias_layer_list[self.cur_iter] = BiasCorrectionLayer().to(self.device)
        
        self.bias_layer = self.bias_layer_list[self.cur_iter]

        if self.val_full and self.cur_iter > 0 and len(self.bias_labels[self.cur_iter]) > 0:
            val_df = pd.DataFrame(self.valid_list)
            
            val_dataset = ImageDataset(
                self.root,
                val_df,
                dataset=self.dataset,
                transform=self.test_transform,
                cls_list=self.exposed_classes,
                data_dir=self.data_dir,
                preload=True,
                n_classes_sup=self.n_classes_sup
            )            

            bias_loader = DataLoader(val_dataset, shuffle=True, batch_size=batch_size, num_workers=n_worker)
            criterion = self.criterion
            
            self.bias_layer = self.bias_layer_list[self.cur_iter]
            optimizer = torch.optim.Adam(params=self.bias_layer.parameters(), lr=0.001)
            self.model.eval()
            total_loss = None
            
            
            model_out_sup = []
            model_out_sub = []
            
            xlabels = []
            hierarchies = []
            for i, data in enumerate(bias_loader):
                x = data["image"]
                xlabel = data["label"]
                hierarchy = data["hierarchy"]
                
                
                x = x.to(self.device)
                xlabel = xlabel.to(self.device)
                hierarchy = hierarchy.to(self.device)
                
                
                with torch.no_grad():
                    out_sup, out_sub = self.model(x)
                    
                    
                model_out_sup.append(out_sup.detach().cpu())
                model_out_sub.append(out_sub.detach().cpu())
                xlabels.append(xlabel.detach().cpu())
                hierarchies.append(hierarchy.detach().cpu())
                
                
            for iteration in range(n_iter):
                self.bias_layer.train()
                total_loss = 0.0
                
                for i, (out_sup, out_sub) in enumerate(zip(model_out_sup,model_out_sub)):
                    logit_sup = self.online_bias_forward(out_sup.to(self.device), self.cur_iter, 0)
                    logit_sub = self.online_bias_forward(out_sub.to(self.device), self.cur_iter, 1)

                    xlabel = xlabels[i].to(self.device)
                    hierarchy = hierarchies[i].to(self.device)
                    
                    idx_sup = (hierarchy == 0)
                    idx_sub = (hierarchy == 1)
                    
                    loss_sup = self.criterion(logit_sup[idx_sup], xlabel[idx_sup])
                    loss_sub = self.criterion(logit_sub[idx_sub], xlabel[idx_sub])                    

                    loss = (loss_sup+loss_sub)/x.size(0)
                    
                    
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    total_loss += loss.item()

                logger.info(
                    "[Stage 2] [{}/{}]\tloss: {:.4f}\talpha: {:.4f}\tbeta: {:.4f}".format(
                        iteration + 1,
                        n_iter,
                        total_loss,
                        self.bias_layer.linear.weight.item(),
                        self.bias_layer.linear.bias.item(),
                    )
                )
            assert total_loss is not None
            
            
            self.print_bias_layer_parameters()

    def distillation_loss(self, old_logit, new_logit):
        # new_logit should have same dimension with old_logit.(dimension = n)
        assert new_logit.size(1) == old_logit.size(1)
        T = 2
        old_softmax = torch.softmax(old_logit / T, dim=1)
        new_log_softmax = torch.log_softmax(new_logit / T, dim=1)
        loss = -(old_softmax * new_log_softmax).sum(dim=1)
        return loss

    def print_bias_layer_parameters(self):
        for i, layer in enumerate(self.bias_layer_list):
            logger.info(
                "[{}] alpha: {:.4f}, beta: {:.4f}".format(
                    i, layer.linear.weight.item(), layer.linear.bias.item()
                )
            )
