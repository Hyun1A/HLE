import os, sys
import logging
import random
import copy
import math
from collections import defaultdict

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
        self.eval_corresponding_super = []        
        
        
        
        #self.models = []
        #self.model = self.model.to('cpu')

    def online_step(self, sample, sample_num, n_worker):
        self.n_count_num +=1
        
        if sample['klass'] not in self.exposed_classes:
            self.add_new_class(sample['klass'], sample['hierarchy'], sample['klass_sup'])
            
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
        self.eval_corresponding_super.append(copy.deepcopy(self.corresponding_super))  
        
        
            
        dummy = {'avg_loss': 0.0, 'avg_acc': 0.0, 'cls_acc': np.zeros(self.n_classes)}   
        return dummy, dummy, dummy      

    def evaluate_all(self, test_list, n_epoch, batch_size, n_worker):
        num_workers = args.num_gpus*args.workers_per_gpu
        num_evals = len(self.eval_samples)
        task_records = defaultdict(list)

        print('num_evals:', num_evals)
        for i in range(math.ceil(num_evals/num_workers)):
            #if self.eval_n_count_num[i*num_workers] < 140000:
                #print(self.eval_n_count_num[i*num_workers])
                #continue
            
            
            workers = [RemoteTrainer.remote(self.root, self.exp_name, self.model_name, self.dataset, self.n_classes, self.n_classes_sup, self.n_classes_sub, self.opt_name, self.lr,
                                            'cos', self.eval_samples[i*num_workers+j], test_list, self.criterion,
                                            self.train_transform, self.test_transform, self.cutmix,
                                            use_amp=self.use_amp, data_dir=self.data_dir, n_count_num=self.eval_n_count_num[i*num_workers+j],
                                            end_task=self.is_end_task[i*num_workers+j], cur_iter=self.iters[i*num_workers+j], mode=self.mode, seed=self.seed, tm=self.tm, memory_size=self.memory_size, 
                                            exposed_classes=self.eval_exposed_classes[i*num_workers+j],
                                            exposed_classes_sup=self.eval_exposed_classes_sup[i*num_workers+j], exposed_classes_sub=self.eval_exposed_classes_sub[i*num_workers+j], 
                                            corresponding_super=self.eval_corresponding_super[i*num_workers+j], check_stream=self.check_stream
                                            )
                                            for j in range(min(num_workers, num_evals-num_workers*i))]
            
            ray.get([workers[j].eval_worker.remote(n_epoch, batch_size, n_worker) for j in range(min(num_workers, num_evals-num_workers*i))])

    
    def after_task(self, cur_iter):
        pass

    

@ray.remote(num_gpus=1 / args.workers_per_gpu)
class RemoteTrainer:
    def __init__(self, root, exp_name, model_name, dataset, n_classes, n_classes_sup, n_classes_sub, opt_name, lr, sched_name, train_list, test_list,
                 criterion, train_transform, test_transform, cutmix, device=0, use_amp=False, data_dir=None, n_count_num=0, end_task=False, cur_iter=0, mode=None, seed=None, tm=None, memory_size=None, \
                 exposed_classes=None, exposed_classes_sup=None, exposed_classes_sub=None, corresponding_super=None, check_stream=0):

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
        
        self.model_name = model_name
        self.dataset = dataset
        self.n_classes = n_classes

        
        self.exposed_classes = exposed_classes
        self.exposed_classes_sup = exposed_classes_sup
        self.exposed_classes_sub = exposed_classes_sub
        self.corresponding_super = corresponding_super
        self.check_stream = check_stream
        

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
        
        self.model = select_model(
            model_name, dataset, self.num_learned_class_sup, self.num_learned_class_sub
        )
        
        self.device = device
        self.model = self.model.cuda(self.device)
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
            n_classes_sup=self.n_classes_sup
        )
        
        test_loader = DataLoader(
            test_dataset,
            shuffle=False,
            batch_size=batch_size,
            num_workers=n_worker,
        )
        
        _,_,_ = self.evaluation(test_loader, self.criterion, self.end_task)
        

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
            n_classes_sup=self.n_classes_sup
        )
        
        
        mem_dataloader = Dataloader(mem_dataset,
                                    shuffle=True,
                                    batch_size=4*batch_size,
                                    num_workers=n_worker,
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
                
            total_loss_sup, correct_sup, num_data_sup = 0.0, 0.0, 0.0
            total_loss_sub, correct_sub, num_data_sub = 0.0, 0.0, 0.0
            
            #idxlist = mem_dataset.generate_idx(4*batch_size)
            for idx, data in enumerate(mem_dataloader):
                #data = mem_dataset.get_data_gpu(idx)
                x = data['image']
                y = data['label'].type(torch.float)
                hierarchy = data['hierarchy'].type(torch.float)

                x = x.to(self.device)
                y = y.to(self.device)
                hierarchy = hierarchy.to(self.device)

                self.optimizer.zero_grad()

                logit_sup, logit_sub, loss_sup, loss_sub = self.model_forward(x, y, hierarchy)
                
                loss = (loss_sup + loss_sub)/x.size(0)
                
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


            print('For superclass')
            print(
                f"Task {cur_iter} | Epoch {epoch + 1}/{n_epoch} | train_loss {train_loss_sup:.4f} | train_acc {train_acc_sup:.4f} | "
                f"lr {self.optimizer.param_groups[0]['lr']:.4f}"
            )
            print('For subclass')
            print(
                f"Task {cur_iter} | Epoch {epoch + 1}/{n_epoch} | train_loss {train_loss_sub:.4f} | train_acc {train_acc_sub:.4f} | "
                f"lr {self.optimizer.param_groups[0]['lr']:.4f}"
            )      
            
    
    def model_forward(self, x, y, hierarchy):
        do_cutmix = self.cutmix and np.random.rand(1) < 0.5
        
        idx_sup = (hierarchy == 0)
        idx_sub = (hierarchy == 1)
        
        if (torch.sum(idx_sup.type(torch.float)) == 1) or (torch.sum(idx_sub.type(torch.float)) == 1):
            do_cutmix=False
        
        loss_sup = torch.tensor([0.]).to(self.device)
        loss_sub = torch.tensor([0.]).to(self.device)
        
        
        if do_cutmix:
            if self.use_amp:
                with torch.cuda.amp.autocast():
                    
                    logit_sup = torch.zeros((x.size(0), len(self.exposed_classes_sup)), dtype=torch.float16).to(self.device)
                    logit_sub = torch.zeros((x.size(0), max(len(self.exposed_classes_sub),1)), dtype=torch.float16).to(self.device)
                    
                    if torch.sum(idx_sup.type(torch.float)) > 1:
                        x_sup, labels_a_sup, labels_b_sup, lam_sup = cutmix_data(x=x[idx_sup], y=y[idx_sup], alpha=1.0)
                        logit_sup_p, logit_sub_p = self.model(x_sup)
                        loss_sup = lam_sup * self.criterion(logit_sup_p, labels_a_sup) + (1 - lam_sup) * self.criterion(logit_sup_p, labels_b_sup)
                        logit_sup[idx_sup] = copy.deepcopy(logit_sup_p.detach())
                        logit_sub[idx_sup] = copy.deepcopy(logit_sub_p.detach())

                    else:
                        loss_sup = torch.tensor([0.]).to(self.device)     

                    
                    if torch.sum(idx_sub.type(torch.float)) > 1:         
                        x_sub, labels_a_sub, labels_b_sub, lam_sub = cutmix_data(x=x[idx_sub], y=y[idx_sub], alpha=1.0)
                        logit_sup_b, logit_sub_b = self.model(x_sub)
                        loss_sub = lam_sub * self.criterion(logit_sub_b, labels_a_sub) + (1 - lam_sub) * self.criterion(logit_sub_b, labels_b_sub)
                        logit_sup[idx_sub] = copy.deepcopy(logit_sup_b.detach())
                        logit_sub[idx_sub] = copy.deepcopy(logit_sub_b.detach())     
                    
                    else:
                        loss_sub = torch.tensor([0.]).to(self.device)
                        
                
        else:
            if self.use_amp:
                with torch.cuda.amp.autocast():
                    logit_sup, logit_sub = self.model(x)
                    
                    if torch.sum(idx_sup.type(torch.float)) != 0:
                        loss_sup = self.criterion(logit_sup[idx_sup], y[idx_sup])
                    if torch.sum(idx_sub.type(torch.float)) != 0:
                        loss_sub = self.criterion(logit_sub[idx_sub], y[idx_sub])
                        

        return logit_sup, logit_sub, loss_sup, loss_sub
            
            
    
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
                    correct_from_sub = (pred_sub[idx_sup]//self.n_classes_sup) == y[idx_sup]
                    
                    total_correct_sup += torch.sum(torch.logical_or(correct_from_sup, correct_from_sub).double()).item()
                    xlabel_cnt_sup, correct_xlabel_cnt_sup = self._interpret_pred(y[idx_sup], pred_sup[idx_sup], pred_sub[idx_sup]//self.n_classes_sup)
                    
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
            folder_name = os.path.join(f"{self.root}/{self.exp_name}/results", self.dataset, self.mode, f'memory_size_{self.memory_size}', f'seed_{self.seed}', 'latest')
        else:
            folder_name = os.path.join(f"{self.root}/{self.exp_name}/results", self.dataset, self.mode, f'memory_size_{self.memory_size}', f'seed_{self.seed}', self.tm)
            
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
