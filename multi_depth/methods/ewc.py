################################
# This code is referred by
# https://github.com/GT-RIPL/Continual-Learning-Benchmark
################################

import logging
import random

import numpy as np
import torch
import torch.nn as nn
import pandas as pd
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from methods.er_baseline import ER
from utils.data_loader import cutmix_data, ImageDataset, StreamDataset
from utils.train_utils import cycle

logger = logging.getLogger()
#writer = SummaryWriter("tensorboard")


class EWCpp(ER):

    def __init__(
        self, criterion, device, train_transform, test_transform, n_classes, n_classes_sup, n_classes_sub, writer, **kwargs
    ):
        super().__init__(
            criterion, device, train_transform, test_transform, n_classes, n_classes_sup, n_classes_sub, writer, **kwargs
        )

        # except for last layers.
        self.params = {
            n: p for n, p in list(self.model.named_parameters())[:-2] if p.requires_grad
        }  # For convenience
        self.regularization_terms = {}
        self.task_count = 0
        self.reg_coef = kwargs["reg_coef"]
        self.online_reg = True

        self.score = []
        self.fisher = []
        self.n_fisher_sample = None
        self.empFI = False
        self.alpha = 0.5
        self.epoch_score = {}
        self.epoch_fisher = {}
        for n, p in self.params.items():
            self.epoch_score[n] = (
                p.clone().detach().fill_(0).to(self.device)
            )  # zero initialized
            self.epoch_fisher[n] = (
                p.clone().detach().fill_(0).to(self.device)
            )  # zero initialized

    def regularization_loss(
        self,
    ):
        reg_loss = 0
        if len(self.regularization_terms) > 0:
            # Calculate the reg_loss only when the regularization_terms exists
            for _, reg_term in self.regularization_terms.items():
                task_reg_loss = 0
                importance = reg_term["importance"]
                task_param = reg_term["task_param"]

                for n, p in self.params.items():
                    task_reg_loss += (importance[n] * (p - task_param[n]) ** 2).sum()

                max_importance = 0
                max_param_change = 0
                for n, p in self.params.items():
                    max_importance = max(max_importance, importance[n].max())
                    max_param_change = max(
                        max_param_change, ((p - task_param[n]) ** 2).max()
                    )
                if reg_loss > 1000:
                    logger.warning(
                        f"max_importance:{max_importance}, max_param_change:{max_param_change}"
                    )
                reg_loss += task_reg_loss
            reg_loss = self.reg_coef * reg_loss

        return reg_loss

    def online_train(self, sample, batch_size, n_worker, iterations=1, stream_batch_size=1):
        self.model.train()
        total_loss_sup, correct_sup, num_data_sup = 0.0, 0.0, 0.0
        total_loss_sub, correct_sub, num_data_sub = 0.0, 0.0, 0.0        
        
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

            old_params = {n: p.clone().detach() for n, p in self.params.items()}
            old_grads = {n: p.grad.clone().detach() for n, p in self.params.items() if p.grad is not None}

            logit_hier, loss_hier = self.model_forward(x, y, hierarchy)
            
            loss = torch.tensor([0.]).to(self.device)
            for l in loss_hier:
                loss+=l
            loss /= x.size(0)

            
            if self.use_amp:
                with torch.cuda.amp.autocast():
                    reg_loss = self.regularization_loss()
                    loss += reg_loss
            else:
                reg_loss = self.regularization_loss()
                loss += reg_loss
            if self.use_amp:
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                self.optimizer.step()
                
            self.update_schedule()
            
            new_params = {n: p.clone().detach() for n, p in self.params.items()}
            new_grads = {
                n: p.grad.clone().detach() for n, p in self.params.items() if p.grad is not None
            }
            self.update_fisher_and_score(new_params, old_params, new_grads, old_grads)
            
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

    def online_after_task(self, cur_iter):
        # 2.Backup the weight of current task
        task_param = {}
        for n, p in self.params.items():
            task_param[n] = p.clone().detach()

        # 3.Calculate the importance of weights for current task
        importance = self.calculate_importance()

        # Save the weight and importance of weights of current task
        self.task_count += 1

        # Use a new slot to store the task-specific information
        if self.online_reg and len(self.regularization_terms) > 0:
            # Always use only one slot in self.regularization_terms
            self.regularization_terms[1] = {
                "importance": importance,
                "task_param": task_param,
            }
        else:
            # Use a new slot to store the task-specific information
            self.regularization_terms[self.task_count] = {
                "importance": importance,
                "task_param": task_param,
            }
        logger.debug(f"# of reg_terms: {len(self.regularization_terms)}")

    def update_fisher_and_score(self, new_params, old_params, new_grads, old_grads, epsilon=0.001):
        for n, _ in self.params.items():
            if n in old_grads:
                new_p = new_params[n]
                old_p = old_params[n]
                new_grad = new_grads[n]
                old_grad = old_grads[n]
                if torch.isinf(new_p).sum()+torch.isinf(old_p).sum()+torch.isinf(new_grad).sum()+torch.isinf(old_grad).sum():
                    continue
                if torch.isnan(new_p).sum()+torch.isnan(old_p).sum()+torch.isnan(new_grad).sum()+torch.isnan(old_grad).sum():
                    continue
                self.epoch_score[n] += (old_grad-new_grad) * (new_p - old_p) / (
                    0.5 * self.epoch_fisher[n] * (new_p - old_p) ** 2 + epsilon
                )
                if self.epoch_score[n].max() > 1000:
                    logger.debug(
                        "Too large score {} / {}".format(
                            (old_grad-new_grad) * (new_p - old_p),
                            0.5 * self.epoch_fisher[n] * (new_p - old_p) ** 2 + epsilon,
                        )
                    )
                if (self.epoch_fisher[n] == 0).all():  # First time
                    self.epoch_fisher[n] = new_grad ** 2
                else:
                    self.epoch_fisher[n] = (1 - self.alpha) * self.epoch_fisher[
                        n
                    ] + self.alpha * new_grad ** 2

    def calculate_importance(self):
        importance = {}
        self.fisher.append(self.epoch_fisher)
        if self.task_count == 0:
            self.score.append(self.epoch_score)
        else:
            score = {}
            for n, p in self.params.items():
                score[n] = 0.5 * self.score[-1][n] + 0.5 * self.epoch_score[n]
            self.score.append(score)

        for n, p in self.params.items():
            importance[n] = self.fisher[-1][n]
            self.epoch_score[n] = self.params[n].clone().detach().fill_(0)
        return importance