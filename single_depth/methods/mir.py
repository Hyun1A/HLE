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

from methods.er_baseline import ER
from utils.data_loader import cutmix_data, ImageDataset, StreamDataset

logger = logging.getLogger()
#writer = SummaryWriter("tensorboard")

class MIR(ER):
    def __init__(
        self, criterion, device, train_transform, test_transform, n_classes, n_classes_sup, n_classes_sub, writer, **kwargs
    ):
        super().__init__(
            criterion, device, train_transform, test_transform, n_classes, n_classes_sup, n_classes_sub, writer, **kwargs
        )        
        
        self.cand_size = kwargs['mir_cands']

    def online_train(self, sample, batch_size, n_worker, iterations=1, stream_batch_size=1):
        self.model.train()
        total_loss_sup, correct_sup, num_data_sup = 0.0, 0.0, 0.0
        total_loss_sub, correct_sub, num_data_sub = 0.0, 0.0, 0.0
        
        
        
        #print(len(self.exposed_classes))
        #print(self.num_learned_class_sup)
        #print(self.num_learned_class_sub)
        #print(sample)
        assert stream_batch_size > 0
        #print()
        
        sample_dataset = StreamDataset(self.root, sample, dataset=self.dataset, transform=self.train_transform,
                                       cls_list=self.exposed_classes, data_dir=self.data_dir, device=self.device,
                                       transform_on_gpu=self.gpu_transform, n_classes_sup=self.n_classes_sup)

        for i in range(iterations):
            stream_data = sample_dataset.get_data()
            str_x = stream_data['image']
            str_y = stream_data['label']
            str_hierarchy = stream_data['hierarchy']
            
            
            x = str_x.to(self.device)
            y = str_y.to(self.device)
            hierarchy = str_hierarchy.to(self.device)
            
            
            logit_sup, logit_sub = self.model(x)
            
            idx_sup = (hierarchy == 0)
            idx_sub = (hierarchy == 1)
            

            loss_sup = self.criterion(logit_sup[idx_sup], y[idx_sup]) \
                if torch.sum(idx_sup.type(torch.float)) != 0 else torch.tensor([0.]).to(self.device)
            
            loss_sub = self.criterion(logit_sub[idx_sub], y[idx_sub]) \
                if torch.sum(idx_sub.type(torch.float)) != 0 else torch.tensor([0.]).to(self.device)
                        
            
            loss = (loss_sup+loss_sub)/x.size(0)
            #print(loss)

            self.optimizer.zero_grad()
            loss.backward()
            grads = {}
            for name, param in self.model.named_parameters():
                grads[name] = param.grad.data if param.grad != None else torch.zeros_like(param).to(self.device)

            if len(self.memory) > 0:
                memory_batch_size = min(len(self.memory), batch_size - stream_batch_size)

                lr = self.optimizer.param_groups[0]['lr']
                new_model = copy.deepcopy(self.model)
                for name, param in new_model.named_parameters():
                    param.data = param.data - lr * grads[name]

                memory_cands, memory_cands_test = self.memory.get_two_batches(min(self.cand_size, len(self.memory)), test_transform=self.test_transform)
                
                x = memory_cands_test['image']
                y = memory_cands_test['label']
                hierarchy = memory_cands_test['hierarchy']
                
                x = x.to(self.device)
                y = y.to(self.device)
                hierarchy = hierarchy.to(self.device)
                
                
                with torch.no_grad():
                    if self.use_amp:
                        with torch.cuda.amp.autocast():
                            logit_pre_sup, logit_pre_sub = self.model(x)
                            logit_post_sup, logit_post_sub = new_model(x)
                            
                            idx_sup = (hierarchy == 0)
                            idx_sub = (hierarchy == 1)       


                            pre_loss_sup = F.cross_entropy(logit_pre_sup[idx_sup], y[idx_sup], reduction='none') \
                                    if torch.sum(idx_sup.type(torch.float)) != 0 else torch.tensor([0.]).to(self.device)
                            post_loss_sup = F.cross_entropy(logit_post_sup[idx_sup], y[idx_sup], reduction='none') \
                                    if torch.sum(idx_sup.type(torch.float)) != 0 else torch.tensor([0.]).to(self.device)

                                
                            pre_loss_sub = F.cross_entropy(logit_pre_sub[idx_sub], y[idx_sub], reduction='none') \
                                    if torch.sum(idx_sub.type(torch.float)) != 0 else torch.tensor([0.]).to(self.device)
                            post_loss_sub = F.cross_entropy(logit_post_sub[idx_sub], y[idx_sub], reduction='none') \
                                    if torch.sum(idx_sub.type(torch.float)) != 0 else torch.tensor([0.]).to(self.device)

                                                        
                            scores_sup = post_loss_sup - pre_loss_sup
                            scores_sub = post_loss_sub - pre_loss_sub
                            
                            
                            
                    else:
                        logit_pre_sup, logit_pre_sub = self.model(x)
                        logit_post_sup, logit_pre_sub = new_model(x)

                        idx_sup = (hierarchy == 0)
                        idx_sub = (hierarchy == 1)       

                        pre_loss_sup = F.cross_entropy(logit_pre_sup[idx_sup], y[idx_sup], reduction='none') \
                                if torch.sum(idx_sup.type(torch.float)) != 0 else torch.tensor([0.]).to(self.device)
                        post_loss_sup = F.cross_entropy(logit_post_sup[idx_sup], y[idx_sup], reduction='none') \
                                if torch.sum(idx_sup.type(torch.float)) != 0 else torch.tensor([0.]).to(self.device)

                        pre_loss_sub = F.cross_entropy(logit_pre_sub[idx_sub], y[idx_sub], reduction='none') \
                                if torch.sum(idx_sub.type(torch.float)) != 0 else torch.tensor([0.]).to(self.device)
                        post_loss_sub = F.cross_entropy(logit_post_sub[idx_sub], y[idx_sub], reduction='none') \
                                if torch.sum(idx_sub.type(torch.float)) != 0 else torch.tensor([0.]).to(self.device)


                        scores_sup = post_loss_sup - pre_loss_sup
                        scores_sub = post_loss_sub - pre_loss_sub
                        
                        
                idx_sup_pos = idx_sup.type(torch.float).nonzero()
                idx_sub_pos = idx_sub.type(torch.float).nonzero()  
                
                ratio_sup = idx_sup_pos.size(0)/(idx_sup_pos.size(0)+idx_sub_pos.size(0))
                
                memory_batch_size_sup = int(ratio_sup*memory_batch_size)
                memory_batch_size_sub = memory_batch_size - memory_batch_size_sup
                
                #print(memory_batch_size_sub)
                #print(scores_sub)
                #print(idx_sub_pos)
                
                selected_samples_sup = idx_sup_pos[ torch.argsort(scores_sup, descending=True)[:memory_batch_size_sup] ]
                selected_samples_sub = idx_sub_pos[ torch.argsort(scores_sub, descending=True)[:memory_batch_size_sub] ]
                selected_samples = torch.cat([selected_samples_sup, selected_samples_sub]).squeeze(-1)
                
                #print('selected_samples', selected_samples.shape)
                
                mem_x = memory_cands['image'][selected_samples]
                mem_y = memory_cands['label'][selected_samples]
                mem_hierarchy = memory_cands['hierarchy'][selected_samples]
                
                
                #print(str_x.shape)
                #print(mem_x.shape)
                #print(str_y.shape)
                #print(mem_y.shape)
                
                #print(str_hierarchy.shape)
                #print(mem_hierarchy.shape)
                
                
                x = torch.cat([str_x, mem_x.squeeze(1)])
                y = torch.cat([str_y, mem_y])
                hierarchy = torch.cat([str_hierarchy, mem_hierarchy])
                
                
                x = x.to(self.device)
                y = y.to(self.device)
                hierarchy = hierarchy.to(self.device)
                

            self.optimizer.zero_grad()
            
            logit_sup, logit_sub, loss_sup, loss_sub = self.model_forward(x, y, hierarchy)
            
            loss = (loss_sup+loss_sub)/x.size(0)
            
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

