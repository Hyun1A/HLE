import os, sys
import logging
import random
import copy
import math
from collections import defaultdict
from copy import deepcopy

from torch.utils.tensorboard import SummaryWriter

import pandas as pd
import numpy as np
import torch
from torch import optim


from methods.er_baseline import ER
from utils.train_utils import select_model, select_optimizer, select_scheduler
from utils.data_loader import ImageDataset, cutmix_data
from torch.utils.data import DataLoader

import ray
from configuration import config

logger = logging.getLogger()
#writer = SummaryWriter("tensorboard")
args = config.base_parser()
if args.mode == 'gdumb':
    ray.init(num_gpus=args.num_gpus)

    
#print(args.workers_per_gpu)
#print(args.num_gpus)

class GDumb(ER):
    def __init__(
        self, criterion, device, train_transform, test_transform, n_classes, n_classes_sup, n_classes_sub, writer, **kwargs
    ):
        super().__init__(
            criterion, device, train_transform, test_transform, n_classes, n_classes_sup, n_classes_sub, writer, **kwargs
        )
        self.memory_epoch = kwargs["memory_epoch"]
        self.memory_size = kwargs["memory_size"]
        self.n_epoch = kwargs["memory_epoch"]
        self.n_worker = kwargs["n_worker"]
        self.batch_size = kwargs["batchsize"]
        self.n_tasks = kwargs["n_tasks"]
        self.eval_period = kwargs["eval_period"]
        self.eval_samples = []
        self.eval_time = []
        self.task_time = []
        
        self.iters = []
        self.eval_n_count_num = []
        self.is_end_task = []
        self.eval_exposed_classes = []
        
        self.check_stream = 0
        self.eval_exposed_classes_sup = []
        self.eval_exposed_classes_sub = []
        self.eval_exposed_classes_hier = [[] for i in range(self.depth+1)]
        self.eval_corresponding_super = []        
        
        
        self.max_hierarchy = 0        
        
        #self.models = []
        #self.model = self.model.to('cpu')

    def online_step(self, sample, sample_num, n_worker):
        self.n_count_num +=1
        
        if sample['klass'] not in self.exposed_classes:
            hierarchy = sample['hierarchy']
            self.add_new_class(sample['klass'], hierarchy, sample[f'klass_{hierarchy}'])
            if hierarchy > self.max_hierarchy:
                self.max_hierarchy = hierarchy               

        self.update_memory(sample)

    def update_memory(self, sample):
        if len(self.memory.images) >= self.memory_size:
            label_frequency = copy.deepcopy(self.memory.cls_count)
            label_frequency[self.exposed_classes.index(sample['klass'])] += 1
            cls_to_replace = np.argmax(np.array(label_frequency))
            idx_to_replace = np.random.choice(self.memory.cls_idx[cls_to_replace])
            self.memory.replace_sample(sample, idx_to_replace)
        else:
            self.memory.replace_sample(sample)


    def online_evaluate(self, test_list, sample_num, batch_size, n_worker, end_task=False):
        
        print('task number:', self.cur_iter)
        print('number of seen data:', self.n_count_num)
        
        self.eval_samples.append(copy.deepcopy(self.memory.datalist))
        self.eval_time.append(sample_num)

        self.eval_n_count_num.append(self.n_count_num)
        self.is_end_task.append(end_task)
        self.iters.append(self.cur_iter)
        self.eval_exposed_classes.append(copy.deepcopy(self.memory.cls_list))
        self.eval_exposed_classes_sup.append(copy.deepcopy(self.exposed_classes_sup))
        self.eval_exposed_classes_sub.append(copy.deepcopy(self.exposed_classes_sub))
        
        
            
        dummy = {'avg_loss': 0.0, 'avg_acc': 0.0, 'cls_acc': np.zeros(self.n_classes)}   
        return dummy, dummy, dummy      

    def evaluate_all(self, test_list, n_epoch, batch_size, n_worker):
        num_workers = args.num_gpus*args.workers_per_gpu
        num_evals = len(self.eval_samples)
        task_records = defaultdict(list)

        print('num_evals:', num_evals)
        for i in range(math.ceil(num_evals/num_workers)):
            if self.eval_n_count_num[i*num_workers] > 50000:
                if self.eval_n_count_num[i*num_workers] <= 25000:
                    #print(self.eval_n_count_num[i*num_workers])
                    continue
            
            
            workers = [RemoteTrainer.remote(self.root, self.exp_name, self.model_name, self.dataset, self.n_classes, self.n_classes_sup, self.n_classes_sub, self.opt_name, self.lr,
                                            'cos', self.eval_samples[i*num_workers+j], test_list, self.criterion,
                                            self.train_transform, self.test_transform, self.cutmix,
                                            use_amp=self.use_amp, data_dir=self.data_dir, n_count_num=self.eval_n_count_num[i*num_workers+j],
                                            end_task=self.is_end_task[i*num_workers+j], cur_iter=self.iters[i*num_workers+j], mode=self.mode, seed=self.seed, tm=self.tm, memory_size=self.memory_size, 
                                            exposed_classes=self.eval_exposed_classes[i*num_workers+j],
                                            exposed_classes_sup=self.eval_exposed_classes_sup[i*num_workers+j], exposed_classes_sub=self.eval_exposed_classes_sub[i*num_workers+j], 
                                            check_stream=self.check_stream, exposed_classes_hier=self.exposed_classes_hier, exposed_hierarchies=self.exposed_hierarchies, depth=self.max_hierarchy, num_learned_class_hier=self.num_learned_class_hier
                                            )
                                            for j in range(min(num_workers, num_evals-num_workers*i))]
            
            ray.get([workers[j].eval_worker.remote(n_epoch, batch_size, n_worker) for j in range(min(num_workers, num_evals-num_workers*i))])

    
    def after_task(self, cur_iter):
        pass

    

@ray.remote(num_gpus=1 / args.workers_per_gpu)
class RemoteTrainer:
    def __init__(self, root, exp_name, model_name, dataset, n_classes, n_classes_sup, n_classes_sub, opt_name, lr, sched_name, train_list, test_list,
                 criterion, train_transform, test_transform, cutmix, device=0, use_amp=False, data_dir=None, n_count_num=0, end_task=False, cur_iter=0, mode=None, seed=None, tm=None, memory_size=None, \
                 exposed_classes=None, exposed_classes_sup=None, exposed_classes_sub=None, check_stream=0, exposed_classes_hier=None,
                exposed_hierarchies=None, depth=0, num_learned_class_hier=None):

        self.root = root
        self.exp_name = exp_name
        
        self.n_classes_sup = n_classes_sup
        self.n_classes_sub = n_classes_sub
        self.n_count_num = n_count_num
        self.end_task = end_task
        self.cur_iter = cur_iter
        
        self.mode = mode
        self.seed = seed
        self.tm = tm
        self.memory_size = memory_size
        self.memory_size_total = self.memory_size
        
        self.model_name = model_name
        self.dataset = dataset
        self.n_classes = n_classes

        
        self.exposed_classes = exposed_classes
        self.exposed_classes_sup = exposed_classes_sup
        self.exposed_classes_sub = exposed_classes_sub
        self.check_stream = check_stream
        
        self.exposed_classes_hier = exposed_classes_hier
        self.exposed_hierarchies = exposed_hierarchies
        self.depth = depth
        self.num_learned_class_hier = num_learned_class_hier
        

        self.train_list = train_list
        
        #print('train_list:', self.train_list)
        
        self.test_list = test_list

        self.train_transform = train_transform
        self.test_transform = test_transform
        self.cutmix = cutmix

        #self.exposed_classes = pd.DataFrame(self.train_list)["klass"].unique().tolist()
        self.exposed_classes = exposed_classes
        self.num_learned_class = len(self.exposed_classes)
        
        
        #self.exposed_classes_sup = [] #0 pd.DataFrame(self.train_list)["klass_sup"].unique().tolist()
        #self.exposed_classes_sub = [] #0 pd.DataFrame(self.train_list)["klass_sub"].unique().tolist()
        self.num_learned_class_sup = 0
        self.num_learned_class_sub = 0
        

        if len(self.exposed_classes) < self.n_classes_sup:
            self.exposed_classes_sup = self.exposed_classes
        else:
            self.exposed_classes_sup = self.exposed_classes[:self.n_classes_sup]
        self.num_learned_class_sup = len(self.exposed_classes_sup)

        if len(self.exposed_classes) > self.n_classes_sup:
            self.exposed_classes_sub = self.exposed_classes[self.n_classes_sup:]
        self.num_learned_class_sub = len(self.exposed_classes_sub)         

        
        self.num_learned_class = max(len(self.exposed_classes),1)
        self.num_learned_class_sup = max(len(self.exposed_classes_sup),1)
        self.num_learned_class_sub = max(len(self.exposed_classes_sub),1)

        """
        print('self.exposed_classes', len(self.exposed_classes))
        print('self.exposed_classes_sup', len(self.exposed_classes_sup))
        print('self.exposed_classes_sub', len(self.exposed_classes_sub))
        
        print('self.n_classes', self.n_classes)
        print('self.n_classes_sup', self.n_classes_sup)
        print('self.n_classes_sub', self.n_classes_sub)
        
        print('self.num_learned_class', self.num_learned_class)
        print('self.num_learned_class_sup', self.num_learned_class_sup)
        print('self.num_learned_class_sub', self.num_learned_class_sub)
        
        input()
        """
        
        #print(self.model_name)
        #print(self.dataset)
        #print(self.depth)
        #print(self.num_learned_class_hier)
        
        self.device = device
        self.model = select_model(self.model_name, self.dataset, num_heads=self.depth+1, \
                                  num_classes_hier = list(self.num_learned_class_hier.cpu().numpy().astype(int))).to(self.device)
        self.criterion = criterion.cuda(self.device)
        self.topk = 1

        self.use_amp = use_amp
        if self.use_amp:
            self.scaler = torch.cuda.amp.GradScaler()

        self.lr = lr
        # Initialize the optimizer and scheduler
        logger.info("Reset the optimizer and scheduler states")
        self.optimizer = select_optimizer(
            opt_name, self.lr, self.model
        )
        self.scheduler = select_scheduler(sched_name, self.optimizer)
        self.data_dir = data_dir

    def reset_opt(self):
        self.optimizer = select_optimizer(self.opt_name, self.lr, self.model)
        self.scheduler = select_scheduler(self.sched_name, self.optimizer, self.lr_gamma)        
        
        
    def eval_worker(self, n_epoch, batch_size, n_worker):
        print('training at ', self.cur_iter)
        
        # online memory train
        self.online_memory_train(cur_iter=self.cur_iter, n_epoch=n_epoch, batch_size=batch_size)
        
        # evaluation
        test_df = pd.DataFrame(self.test_list)
        
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
        
        _ = self.evaluation(test_loader, self.criterion, self.end_task)
        

    def online_memory_train(self, cur_iter, n_epoch, batch_size):
        if self.dataset == 'imagenet':
            self.scheduler = optim.lr_scheduler.MultiStepLR(
                self.optimizer, milestones=[30, 60, 80, 90], gamma=0.1
            )
        else:
            self.scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
                self.optimizer, T_0=1, T_mult=2, eta_min=self.lr * 0.01
            )
            
        #print(self.n_classes_sup)
            
        mem_dataset = ImageDataset(
            self.root,
            pd.DataFrame(self.train_list),
            dataset=self.dataset,
            transform=self.train_transform,
            cls_list=self.exposed_classes,
            data_dir=self.data_dir,
            preload=True,
            device=self.device,
            transform_on_gpu=True,
            n_classes_sup=self.n_classes_sup,
            hierarchy_list=self.exposed_hierarchies,
            depth=self.depth
        )
        
        
        
        for epoch in range(n_epoch):
            print('Epoch:', epoch)
            
            if epoch <= 0:  # Warm start of 1 epoch
                for param_group in self.optimizer.param_groups:
                    param_group["lr"] = self.lr * 0.1
            elif epoch == 1:  # Then set to maxlr
                for param_group in self.optimizer.param_groups:
                    param_group["lr"] = self.lr
            else:  # Aand go!
                self.scheduler.step()
                
            total_loss_hier, correct_hier, num_data_hier = np.zeros(self.depth+1), np.zeros(self.depth+1), np.zeros(self.depth+1)

            
            idxlist = mem_dataset.generate_idx(batch_size)
            for idx in idxlist:
                data = mem_dataset.get_data_gpu(idx)
                x = data["image"]
                y = data["label"]
                hierarchy = data["hierarchy"]
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
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 10)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 10)
                    self.optimizer.step()
                    
                preds_hier = []
                for logit in logit_hier:
                    _, preds = logit.topk(self.topk, 1, True, True)
                    preds_hier.append(preds)

                idx_hier = []
                for h in range(self.depth+1):
                    idx_hier.append((hierarchy==h))

                for h in range(self.depth+1):
                    total_loss_hier[h] += loss_hier[h].item()
                    
                    #print(h)
                    #print(preds_hier)
                    #print(y)
                    #print(idx_hier)
                    #print()
                    
                    correct_hier[h] += torch.sum(preds_hier[h][idx_hier[h]] == y[idx_hier[h]].unsqueeze(1)).item()
                    num_data_hier[h] += y[idx_hier[h]].size(0)           

            train_loss_hier = []
            train_acc_hier = []
            for h in range(self.depth+1):
                train_loss_hier.append( total_loss_hier[h] / num_data_hier[h] if num_data_hier[h] != 0 else 0. )
                train_acc_hier.append( correct_hier[h] / num_data_hier[h] if num_data_hier[h] != 0 else 0. )


                
            for h in range(self.depth+1):
                print(f'hierarchy {h}')
                
                print(
                    f"Task {cur_iter} | Epoch {epoch + 1}/{n_epoch} | train_loss {train_loss_hier[h]:.4f} | train_acc {train_acc_hier[h]:.4f} | "
                    f"lr {self.optimizer.param_groups[0]['lr']:.4f}"
                )                


    
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
                        #print(self.model.fc)
                        #print(h)
                        #print(logit_hier[h][idx_hier[h]])
                        #print(y[idx_hier[h]])
                        
                        
                        loss_hier.append( 
                                          self.criterion(logit_hier[h][idx_hier[h]], y[idx_hier[h]]) \
                                          if torch.sum(idx_hier[h].type(torch.float)) != 0 else torch.tensor([0.]).to(self.device)
                                        )

        #print(logit_hier)
        #input()
                        
        return logit_hier, loss_hier
            
            
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
            
        return ret


    def _interpret_pred(self, y, pred, pred2=None):
        # xlable is batch
        ret_num_data = torch.zeros(self.n_classes)
        ret_corrects = torch.zeros(self.n_classes)

        xlabel_cls, xlabel_cnt = y.unique(return_counts=True)
        for cls_idx, cnt in zip(xlabel_cls, xlabel_cnt):
            ret_num_data[cls_idx] = cnt

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
            
