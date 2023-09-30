import logging
import copy

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from utils.augment import Cutout, Invert, Solarize, select_autoaugment
from torchvision import transforms
from randaugment.randaugment import RandAugment

from methods.er_baseline import ER
from utils.data_loader import cutmix_data, ImageDataset
from utils.augment import Cutout, Invert, Solarize, select_autoaugment

logger = logging.getLogger()
#writer = SummaryWriter("tensorboard")


def cycle(iterable):
    # iterate with shuffling
    while True:
        for i in iterable:
            yield i


class RM(ER):
    def __init__(
        self, criterion, device, train_transform, test_transform, n_classes, n_classes_sup, n_classes_sub, writer, **kwargs
    ):
        super().__init__(
            criterion, device, train_transform, test_transform, n_classes, n_classes_sup, n_classes_sub, writer, **kwargs
        )
        #self.sched_name = "const"
        self.batch_size = kwargs["batchsize"]
        self.memory_epoch = kwargs["memory_epoch"]
        self.n_worker = kwargs["n_worker"]
        self.data_cnt = 0

    def online_step(self, sample, sample_num, n_worker):
        self.n_count_num +=1

        
        if sample['klass'] not in self.exposed_classes:
            self.add_new_class(sample['klass'], sample['hierarchy'], sample['klass_sup'])

        self.temp_batch.append(sample)
        self.num_updates += self.online_iter

        if len(self.temp_batch) == self.batch_size:
            train_loss_sup, train_acc_sup, train_loss_sub, train_acc_sub = \
            self.online_train(self.temp_batch, self.batch_size, n_worker, iterations=int(self.num_updates),  stream_batch_size=self.batch_size)
            
            print('super class')
            self.report_training(sample_num, train_loss_sup, train_acc_sup)
            print('sub class')
            self.report_training(sample_num, train_loss_sub, train_acc_sub)                
            

            for stored_sample in self.temp_batch:
                self.update_memory(stored_sample)
            self.temp_batch = []
            self.num_updates -= int(self.num_updates)
    """
    def add_new_class(self, class_name, hierarchy):
        self.exposed_classes.append(class_name)
        self.num_learned_class = len(self.exposed_classes)
        
        if hierarchy == 0:
            self.num_learned_class_sup += 1 
            self.model.fc_sup = nn.Linear(self.model.fc_sup.in_features, self.num_learned_class_sup).to(self.device)
        
        else:
            self.num_learned_class_sub += 1 
            self.model.fc_sub = nn.Linear(self.model.fc_sub.in_features, self.num_learned_class_sub).to(self.device)
        
        self.memory.add_new_class(cls_list=self.exposed_classes)
        self.reset_opt()
    """
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

    def update_memory(self, sample):
        if len(self.memory.images) >= self.memory_size:
            label_frequency = copy.deepcopy(self.memory.cls_count)
            label_frequency[self.exposed_classes.index(sample['klass'])] += 1
            cls_to_replace = np.argmax(np.array(label_frequency))
            idx_to_replace = np.random.choice(self.memory.cls_idx[cls_to_replace])
            self.memory.replace_sample(sample, idx_to_replace)
        else:
            self.memory.replace_sample(sample)

    def online_before_task(self, cur_iter):
        self.reset_opt()
        #self.scheduler = optim.lr_scheduler.LambdaLR(self.optimizer, lambda iter: 1)

    def online_after_task(self, cur_iter):
        self.reset_opt()
        self.online_memory_train(
            cur_iter=cur_iter,
            n_epoch=self.memory_epoch,
            batch_size=self.batch_size,
        )

    def online_memory_train(self, cur_iter, n_epoch, batch_size):
        if self.dataset == 'imagenet':
            self.scheduler = optim.lr_scheduler.MultiStepLR(
                self.optimizer, milestones=[30, 60, 80, 90], gamma=0.1
            )
        else:
            self.scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
                self.optimizer, T_0=1, T_mult=2, eta_min=self.lr * 0.01
            )
        mem_dataset = ImageDataset(
            self.root,
            pd.DataFrame(self.memory.datalist),
            dataset=self.dataset,
            transform=self.train_transform,
            cls_list=self.exposed_classes,
            data_dir=self.data_dir,
            preload=True,
            device=self.device,
            transform_on_gpu=True,
            n_classes_sup=self.n_classes_sup
        )
        
        for epoch in range(n_epoch):
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
            logger.info(
                f"Task {cur_iter} | Epoch {epoch + 1}/{n_epoch} | train_loss {train_loss_sup:.4f} | train_acc {train_acc_sup:.4f} | "
                f"lr {self.optimizer.param_groups[0]['lr']:.4f}"
            )
            print('For subclass')
            logger.info(
                f"Task {cur_iter} | Epoch {epoch + 1}/{n_epoch} | train_loss {train_loss_sub:.4f} | train_acc {train_acc_sub:.4f} | "
                f"lr {self.optimizer.param_groups[0]['lr']:.4f}"
            )            

    def uncertainty_sampling(self, samples, num_class):
        """uncertainty based sampling

        Args:
            samples ([list]): [training_list + memory_list]
        """
        self.montecarlo(samples, uncert_metric="vr_randaug")

        sample_df = pd.DataFrame(samples)
        mem_per_cls = self.memory_size // num_class

        ret = []
        for i in range(num_class):
            cls_df = sample_df[sample_df["label"] == i]
            if len(cls_df) <= mem_per_cls:
                ret += cls_df.to_dict(orient="records")
            else:
                jump_idx = len(cls_df) // mem_per_cls
                uncertain_samples = cls_df.sort_values(by="uncertainty")[::jump_idx]
                ret += uncertain_samples[:mem_per_cls].to_dict(orient="records")

        num_rest_slots = self.memory_size - len(ret)
        if num_rest_slots > 0:
            logger.warning("Fill the unused slots by breaking the equilibrium.")
            try:
                ret += (
                    sample_df[~sample_df.file_name.isin(pd.DataFrame(ret).file_name)]
                    .sample(n=num_rest_slots)
                    .to_dict(orient="records")
                )
            except:
                ret += (
                    sample_df[~sample_df.filepath.isin(pd.DataFrame(ret).filepath)]
                        .sample(n=num_rest_slots)
                        .to_dict(orient="records")
                )

        try:
            num_dups = pd.DataFrame(ret).file_name.duplicated().sum()
        except:
            num_dups = pd.DataFrame(ret).filepath.duplicated().sum()
        if num_dups > 0:
            logger.warning(f"Duplicated samples in memory: {num_dups}")

        return ret

    def _compute_uncert(self, infer_list, infer_transform, uncert_name):
        batch_size = 32
        infer_df = pd.DataFrame(infer_list)
        infer_dataset = ImageDataset(
            self.root, infer_df, dataset=self.dataset, transform=infer_transform, data_dir=self.data_dir
        )
        infer_loader = DataLoader(
            infer_dataset, shuffle=False, batch_size=batch_size, num_workers=2
        )

        self.model.eval()
        with torch.no_grad():
            for n_batch, data in enumerate(infer_loader):
                x = data["image"]
                x = x.to(self.device)
                logit = self.model(x)
                logit = logit.detach().cpu()

                for i, cert_value in enumerate(logit):
                    sample = infer_list[batch_size * n_batch + i]
                    sample[uncert_name] = 1 - cert_value

    def montecarlo(self, candidates, uncert_metric="vr"):
        transform_cands = []
        logger.info(f"Compute uncertainty by {uncert_metric}!")
        if uncert_metric == "vr":
            transform_cands = [
                Cutout(size=8),
                Cutout(size=16),
                Cutout(size=24),
                Cutout(size=32),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.RandomRotation(45),
                transforms.RandomRotation(90),
                Invert(),
                Solarize(v=128),
                Solarize(v=64),
                Solarize(v=32),
            ]
        elif uncert_metric == "vr_randaug":
            for _ in range(12):
                transform_cands.append(RandAugment())
        elif uncert_metric == "vr_cutout":
            transform_cands = [Cutout(size=16)] * 12
        elif uncert_metric == "vr_autoaug":
            transform_cands = [select_autoaugment(self.dataset)] * 12

        n_transforms = len(transform_cands)

        for idx, tr in enumerate(transform_cands):
            _tr = transforms.Compose([tr] + self.test_transform.transforms)
            self._compute_uncert(candidates, _tr, uncert_name=f"uncert_{str(idx)}")

        for sample in candidates:
            self.variance_ratio(sample, n_transforms)

    def variance_ratio(self, sample, cand_length):
        vote_counter = torch.zeros(sample["uncert_0"].size(0))
        for i in range(cand_length):
            top_class = int(torch.argmin(sample[f"uncert_{i}"]))  # uncert argmin.
            vote_counter[top_class] += 1
        assert vote_counter.sum() == cand_length
        sample["uncertainty"] = (1 - vote_counter.max() / cand_length).item()

    def equal_class_sampling(self, samples, num_class):
        mem_per_cls = self.memory_size // num_class
        sample_df = pd.DataFrame(samples)
        # Warning: assuming the classes were ordered following task number.
        ret = []
        for y in range(self.num_learned_class):
            cls_df = sample_df[sample_df["label"] == y]
            ret += cls_df.sample(n=min(mem_per_cls, len(cls_df))).to_dict(
                orient="records"
            )

        num_rest_slots = self.memory_size - len(ret)
        if num_rest_slots > 0:
            logger.warning("Fill the unused slots by breaking the equilibrium.")
            try:
                ret += (
                    sample_df[~sample_df.file_name.isin(pd.DataFrame(ret).file_name)]
                    .sample(n=num_rest_slots)
                    .to_dict(orient="records")
                )
            except:
                ret += (
                    sample_df[~sample_df.filepath.isin(pd.DataFrame(ret).filepath)]
                        .sample(n=num_rest_slots)
                        .to_dict(orient="records")
                )

        try:
            num_dups = pd.DataFrame(ret).file_name.duplicated().sum()
        except:
            num_dups = pd.DataFrame(ret).filepath.duplicated().sum()
        if num_dups > 0:
            logger.warning(f"Duplicated samples in memory: {num_dups}")

        return ret