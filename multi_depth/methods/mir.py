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
        
        total_loss_hier, correct_hier, num_data_hier = np.zeros(self.depth+1), np.zeros(self.depth+1), np.zeros(self.depth+1)
        
        
        #print(len(self.exposed_classes))
        #print(self.num_learned_class_sup)
        #print(self.num_learned_class_sub)
        #print(sample)
        assert stream_batch_size > 0
        #print()
        
        sample_dataset = StreamDataset(self.root, sample, dataset=self.dataset, transform=self.train_transform,
                                       cls_list=self.exposed_classes, data_dir=self.data_dir, device=self.device,
                                       transform_on_gpu=self.gpu_transform, n_classes_sup=self.n_classes_sup,
                                       hierarchy_list=self.exposed_hierarchies, depth=self.depth)
        
        for i in range(iterations):
            stream_data = sample_dataset.get_data()
            str_x = stream_data['image']
            str_y = stream_data['label']
            str_hierarchy = stream_data['hierarchy']
            
            
            x = str_x.to(self.device)
            y = str_y.to(self.device)
            hierarchy = str_hierarchy.to(self.device)
            
            
            logit_hier = self.model(x)
            
            idx_hier = []
            for h in range(self.depth+1):
                idx_hier.append((hierarchy==h))
            
  
            loss_hier = []   
            for h in range(self.depth+1):
                loss_hier.append( 
                                  self.criterion(logit_hier[h][idx_hier[h]], y[idx_hier[h]]) \
                                  if torch.sum(idx_hier[h].type(torch.float)) != 0 else torch.tensor([0.]).to(self.device)
                                )
                
            
            loss = torch.tensor([0.]).to(self.device)
            for l in loss_hier:
                loss+=l
            loss /= x.size(0)
            
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
                            logit_pre_hier = self.model(x)
                            logit_post_hier = new_model(x)
                            
                            idx_hier = []
                            for h in range(self.depth+1):
                                idx_hier.append((hierarchy==h))

                                
                            pre_loss_hier = []
                            for h in range(self.depth+1):
                                pre_loss_hier.append(  F.cross_entropy(logit_pre_hier[h][idx_hier[h]], y[idx_hier[h]], reduction='none') \
                                        if torch.sum(idx_hier[h].type(torch.float)) != 0 else torch.tensor([0.]).to(self.device) )
                            
                            post_loss_hier = []
                            for h in range(self.depth+1):
                                post_loss_hier.append( F.cross_entropy(logit_post_hier[h][idx_hier[h]], y[idx_hier[h]], reduction='none') \
                                        if torch.sum(idx_hier[h].type(torch.float)) != 0 else torch.tensor([0.]).to(self.device) )

                            scores_hier = []
                            for h in range(self.depth+1):
                                scores_hier.append( post_loss_hier[h] - pre_loss_hier[h] )
                            
                            
                            
                    else:
                        logit_pre_hier = self.model(x)
                        logit_post_hier = new_model(x)

                        idx_hier = []
                        for h in range(self.depth+1):
                            idx_hier.append((hierarchy==h))

                        pre_loss_hier = []
                        for h in range(self.depth+1):
                            pre_loss_hier.append(  F.cross_entropy(logit_pre_hier[h][idx_hier[h]], y[idx_hier[h]], reduction='none') \
                                    if torch.sum(idx_hier[h].type(torch.float)) != 0 else torch.tensor([0.]).to(self.device) )

                        post_loss_hier = []
                        for h in range(self.depth+1):
                            post_loss_hier.append( F.cross_entropy(logit_post_hier[h][idx_hier[h]], y[idx_hier[h]], reduction='none') \
                                    if torch.sum(idx_hier[h].type(torch.float)) != 0 else torch.tensor([0.]).to(self.device) )

                        scores_hier = []
                        for h in range(self.depth+1):
                            scores_hier.append( post_loss_hier[h] - pre_loss_hier[h] )

                idx_hier_pos = []
                for h in range(self.depth+1):
                    idx_hier_pos.append( idx_hier[h].type(torch.float).nonzero() )
                
                len_cands_data = hierarchy.size(0)
                num_data = 0
                
                ratio = []
                memory_batch_size_hier = []
                for h in range(self.depth+1):
                    ratio.append( idx_hier_pos[h].size(0)/len_cands_data )
                    memory_batch_size_hier.append( int(ratio[h]*memory_batch_size) )
                    num_data += memory_batch_size_hier[h]
                
                #memory_batch_size_hier.append( memory_batch_size - num_data )
                for h in range(self.depth+1):
                    if num_data == memory_batch_size:
                        break
                    else:
                        if ratio*memory_batch_size != 0.:
                            memory_batch_size_hier[h] += 1
                            num_data += 1
                
                selected_samples_hier = []
                for h in range(self.depth+1):
                    #print(f'idx_hier_pos[{h}]:', idx_hier_pos[h])
                    #print(f'scores_hier[{h}]:', scores_hier[h])
                    #print(f'memory_batch_size_hier[{h}]:',memory_batch_size_hier[h] )
                    
                    selected_samples_hier.append( idx_hier_pos[h][ torch.argsort(scores_hier[h], descending=True)[:memory_batch_size_hier[h]] ] )
                
                selected_samples = torch.cat(selected_samples_hier).squeeze(-1)
                
                mem_x = memory_cands['image'][selected_samples]
                mem_y = memory_cands['label'][selected_samples]
                mem_hierarchy = memory_cands['hierarchy'][selected_samples]
                
                
                x = torch.cat([str_x, mem_x.squeeze(1)])
                y = torch.cat([str_y, mem_y])
                hierarchy = torch.cat([str_hierarchy, mem_hierarchy])
                
                
                x = x.to(self.device)
                y = y.to(self.device)
                hierarchy = hierarchy.to(self.device)
                

            self.optimizer.zero_grad()
            
            logit_hier, loss_hier = self.model_forward(x, y, hierarchy)
            
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
    
