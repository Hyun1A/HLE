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
            self.valid_size = round(self.memory_size * 0.2)
        elif 'imagenet_subset' in self.dataset:
            self.valid_size = round(self.memory_size * 0.1)
        elif 'cub_200_2011' in self.dataset:
            self.valid_size = round(self.memory_size * 0.1)
        elif 'inat19' in self.dataset:
            self.valid_size = round(self.memory_size * 0.2)
            
            
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
        self.bias_labels_hier = [[] for h in range(self.depth+1)]
        

    def online_before_task(self, cur_iter):
        super().online_before_task(cur_iter)
        self.cur_iter = cur_iter
        self.bias_labels.append([])
        
        for h in range(self.depth+1):
            self.bias_labels_hier[h].append([])
        

        
    def online_after_task(self, cur_iter):
        if self.distilling:
            self.prev_model = deepcopy(self.model)

    def online_step(self, sample, sample_num, n_worker):
        self.n_count_num +=1
        
        if sample['klass'] not in self.exposed_classes:
            hierarchy = sample['hierarchy']
            self.add_new_class(sample['klass'], hierarchy, sample[f'klass_{hierarchy}'])

        use_sample = self.online_valid_update(sample)
        
        
        if use_sample:
            self.num_updates += self.online_iter
            self.temp_batch.append(sample)
            if len(self.temp_batch) == self.temp_batchsize:
                train_loss_hier, train_acc_hier = self.online_train(self.temp_batch, self.batch_size, n_worker, \
                                                      iterations=int(self.num_updates), stream_batch_size=self.temp_batchsize)

                for h in range(self.depth+1):
                    print(f'hierarchy {h}')
                    self.report_training(sample_num, train_loss_hier[h], train_acc_hier[h])
                for h in range(self.depth+1):
                    print(self.model.fc[h])
                    
                print()    
                
                for stored_sample in self.temp_batch:
                    self.update_memory(stored_sample)
                self.temp_batch = []
                self.num_updates = 0

    def add_new_class(self, class_name, hierarchy, class_name_sup=None, class_name_sub=None):
        if self.distilling:
            self.prev_model = deepcopy(self.model)
            
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

        self.bias_labels_hier[hierarchy][self.cur_iter].append(int(self.num_learned_class_hier[hierarchy] - 1))    
        self.bias_labels[self.cur_iter].append(self.num_learned_class - 1)
        
        
        if self.num_learned_class > 1:
            self.online_reduce_valid(self.num_learned_class)

        if 'reset' in self.sched_name:
            self.update_schedule(reset=True)

    
    def online_reduce_valid(self, num_learned_class):
        self.val_per_cls = self.valid_size//num_learned_class
        val_df = pd.DataFrame(self.valid_list)

        
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

        total_loss_hier, correct_hier, num_data_hier = np.zeros(self.depth+1), np.zeros(self.depth+1), np.zeros(self.depth+1)
        distill_loss_hier, classify_loss_hier = np.zeros(self.depth+1), np.zeros(self.depth+1)
        
        if stream_batch_size > 0:
            sample_dataset = StreamDataset(self.root, sample, dataset=self.dataset, transform=self.train_transform, \
                                           cls_list=self.exposed_classes, data_dir=self.data_dir, device=self.device, \
                                           transform_on_gpu=self.gpu_transform, n_classes_sup=self.n_classes_sup,\
                                           hierarchy_list=self.exposed_hierarchies, depth=self.depth)
            
            
            
            
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
            
            idx_hier = []
            num_idx_hier = []

            for h in range(self.depth+1):
                idx_hier.append((hierarchy==h))
                num_idx_hier.append(  torch.sum(idx_hier[h].type(torch.float))  )
                
            
            if do_cutmix:
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
                
                if self.cur_iter != 0:
                    if self.distilling:
                        with torch.no_grad():
                            if self.use_amp:
                                with torch.cuda.amp.autocast():
                                    logit_old_hier = self.prev_model(x)
                                    
                                    for h in range(self.depth+1):
                                        logit_old_hier[h] = self.online_bias_forward(logit_old_hier[h], self.cur_iter-1, h)
                             
                            
                self.optimizer.zero_grad()
                            
                if self.use_amp:
                    with torch.cuda.amp.autocast():
                        logit_new_hier = self.model(x)
                        
                        loss_c_hier = []   
                        for h in range(self.depth+1):
                            loss_c_hier.append( 
                                              lam_hier[h]*self.criterion(logit_new_hier[h][idx_hier[h]], labels_a[idx_hier[h]]) + \
                                              (1 - lam_hier[h])*self.criterion(logit_new_hier[h][idx_hier[h]], labels_b[idx_hier[h]]) \
                                              if torch.sum(idx_hier[h].type(torch.float)) != 0 else torch.tensor([0.]).to(self.device)
                                            )                    
                        
                        loss_c = torch.tensor([0.]).to(self.device)
                        for l in loss_c_hier:
                            loss_c+=l
                        
            else:
                if self.cur_iter != 0:
                    if self.distilling:
                        with torch.no_grad():
                            if self.use_amp:
                                with torch.cuda.amp.autocast():
                                    logit_old_hier = self.prev_model(x)
                                    
                                    for h in range(self.depth+1):
                                        logit_old_hier[h] = self.online_bias_forward(logit_old_hier[h], self.cur_iter-1, h)

                                    
                self.optimizer.zero_grad()
                                    
                if self.use_amp:
                    with torch.cuda.amp.autocast():
                        logit_new_hier = self.model(x)
                        
                        loss_c_hier = []   
                        
                        for h in range(self.depth+1):
                            loss_c_hier.append( 
                                              self.criterion(logit_new_hier[h][idx_hier[h]], y[idx_hier[h]]) \
                                              if torch.sum(idx_hier[h].type(torch.float)) != 0 else torch.tensor([0.]).to(self.device)
                                            )                        
                                        
                        loss_c = torch.tensor([0.]).to(self.device)
                        for l in loss_c_hier:
                            loss_c+=l
                    

            if self.distilling:
                loss_d = torch.tensor(0.0).to(self.device)
                if self.cur_iter == 0:
                    loss_d_hier = [torch.tensor(0.0).to(self.device) for h in range(self.depth+1)]
                    
                else:
                    loss_d_hier = []
                    for h in range(self.depth+1):
                        loss_d_hier.append( self.distillation_loss(logit_old_hier[h], logit_new_hier[h][:, : logit_old_hier[h].size(1)]).sum()   )
                                        
                    for l in loss_d_hier:
                        loss_d+=l

            else:
                loss_d = torch.tensor(0.0).to(self.device)
                loss_d_hier = [torch.tensor(0.0).to(self.device) for h in range(self.depth+1)]


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
            
            preds_hier = []
            for logit in logit_new_hier:
                _, preds = logit.topk(self.topk, 1, True, True)
                preds_hier.append(preds)
            
            idx_hier = []
            for h in range(self.depth+1):
                idx_hier.append((hierarchy==h))

            for h in range(self.depth+1):
                total_loss_hier[h] += (loss_c_hier[h]+loss_d_hier[h]).item()
                correct_hier[h] += torch.sum(preds_hier[h][idx_hier[h]] == y[idx_hier[h]].unsqueeze(1)).item()
                num_data_hier[h] += y[idx_hier[h]].size(0)           

        train_loss_hier = []
        train_acc_hier = []
        for h in range(self.depth+1):
            train_loss_hier.append( total_loss_hier[h] / num_data_hier[h] if num_data_hier[h] != 0 else 0. )
            train_acc_hier.append( correct_hier[h] / num_data_hier[h] if num_data_hier[h] != 0 else 0. )
            
        return train_loss_hier, train_acc_hier
    

    def online_bias_forward(self, input, iter, hierarchy):
        bias_labels = self.bias_labels_hier[hierarchy][iter]   
        bias_layer = self.bias_layer_list[iter]
        
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
        self.bias_layer.eval()
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
                for h in range(self.depth+1):
                    logit_hier[h] = self.online_bias_forward(logit_hier[h], self.cur_iter - 1, h)                

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
                n_classes_sup=self.n_classes_sup,
                hierarchy_list=self.exposed_hierarchies,
                depth=self.depth
            )            

            bias_loader = DataLoader(val_dataset, shuffle=True, batch_size=batch_size, num_workers=n_worker)
            criterion = self.criterion
            
            self.bias_layer = self.bias_layer_list[self.cur_iter]
            optimizer = torch.optim.Adam(params=self.bias_layer.parameters(), lr=0.001)
            self.model.eval()
            total_loss = None
            
            model_out_hier = []
            
            
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
                    out_hier = self.model(x)
                    
                model_out_hier.append([out.detach().cpu() for out in out_hier])

                xlabels.append(xlabel.detach().cpu())
                hierarchies.append(hierarchy.detach().cpu())
                
                
            for iteration in range(n_iter):
                self.bias_layer.train()
                total_loss = 0.0
                
                for i, out_hier in enumerate(model_out_hier):
                    logit_hier = []
                    for h in range(self.depth+1):
                        logit_hier.append( self.online_bias_forward(out_hier[h].to(self.device), self.cur_iter-1, h) )                 
                    
                    xlabel = xlabels[i].to(self.device)
                    hierarchy = hierarchies[i].to(self.device)
                    
                    idx_hier = []
                    for h in range(self.depth+1):
                        idx_hier.append((hierarchy==h))
                    
                    loss_hier = []
                    for h in range(self.depth+1):
                        loss_hier.append( 
                                          self.criterion(logit_hier[h][idx_hier[h]], xlabel[idx_hier[h]]) \
                                          if torch.sum(idx_hier[h].type(torch.float)) != 0 else torch.tensor([0.]).to(self.device)
                                        )          
                                     
                    loss = torch.tensor([0.]).to(self.device)
                    for l in loss_hier:
                        loss+=l
                    loss /= x.size(0)

                    
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
