import logging.config
import os
from typing import List
import time

import PIL
import numpy as np
import pandas as pd
import torch
from torchvision import transforms
from torchvision.transforms.functional import to_pil_image
from torch.utils.data import Dataset
from time import perf_counter

logger = logging.getLogger()

#ROOT_PATH = '/home/sonic/cvpr2023_cl_2/datasets/cifar100/data'
#ROOT_PATH = '/data/cvpr2023_cl_2/datasets/imagenet/data'
ROOT_PATH = '/data/cvpr2023_cl_2/datasets/cifar100/data'

#ROOT_PATH = '/home/sonic/cvpr2023_cl_2/datasets/imagenet/data'


class ImageDataset(Dataset):
    def __init__(self, root, data_frame: pd.DataFrame, dataset: str, transform=None, cls_list=None, data_dir=None,
                 preload=False, device=None, transform_on_gpu=False, n_classes_sup=None):
        
        raw_dataset = "."
        if 'cifar100' in dataset:
            raw_data='cifar100'
        elif 'stanford_car' in dataset:
            raw_data='stanford_car'
        elif 'imagenet' in dataset:
            raw_data='imagenet'
        else:
            raise Exception('Invalid dataset')     
            
        self.root_path = f"{root}/datasets/{raw_data}/data"
        self.n_classes_sup = n_classes_sup
        
        
        self.data_frame = data_frame
        self.dataset = dataset
        self.transform = transform
        self.cls_list = cls_list
        self.data_dir = data_dir
        self.preload = preload
        self.device = device
        self.transform_on_gpu = transform_on_gpu
        self.loaded_images = []
        

        if self.preload:
            mean, std, n_classes, n_classes_sup, n_classes_sub, inp_size, _ = get_statistics(dataset=self.dataset)
            if self.transform_on_gpu:
                self.transform_cpu = transforms.Compose(
                    [
                        transforms.Resize((inp_size, inp_size)),
                        transforms.PILToTensor()
                    ])
                self.transform_gpu = self.transform
            self.loaded_images = []

            
            for idx in range(len(self.data_frame)):
                sample = dict()
                
                try:
                    img_name = self.data_frame.iloc[idx]["file_name"]
                except KeyError:
                    img_name = self.data_frame.iloc[idx]["filepath"]
                
                dir_names = img_name.split('/')
                idx_=dir_names.index('data')
                dir_names = dir_names[(idx_+1):]
                img_name = self.root_path
                for n in dir_names:
                    img_name = os.path.join(img_name, n)

                
                if self.cls_list is None:
                    label = self.data_frame.iloc[idx].get("label", -1)
                else:
                    label = self.cls_list.index(self.data_frame.iloc[idx]["klass"])
                if self.data_dir is None:
                    img_path = os.path.join("dataset", self.dataset, img_name)
                else:
                    img_path = os.path.join(self.data_dir, img_name)
                image = PIL.Image.open(img_path).convert("RGB")
                if self.transform_on_gpu:
                    image = self.transform_cpu(PIL.Image.open(img_path).convert('RGB'))
                elif self.transform:
                    image = self.transform(image)
                    
                    
                hierarchy = self.data_frame.iloc[idx]["hierarchy"]    
                cls_sup = self.data_frame.iloc[idx]["label_sup"]    
                cls_sub = self.data_frame.iloc[idx]["label_sub"]                        
                    
                
                sample["image"] = image
                sample["label"] = label
                sample["image_name"] = img_name
                
                sample["hierarchy"] = hierarchy
                sample["cls_sup"] = cls_sup
                sample["cls_sub"] = cls_sub                
                

                if sample["hierarchy"] == 1:
                    sample["label"] -= self.n_classes_sup                
                            
                
                
                self.loaded_images.append(sample)

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        if self.preload:
            return self.loaded_images[idx]
        else:

            sample = dict()
            if torch.is_tensor(idx):
                idx = idx.tolist()
            try:
                img_name = self.data_frame.iloc[idx]["file_name"]
            except KeyError:
                img_name = self.data_frame.iloc[idx]["filepath"]
            
            dir_names = img_name.split('/')
            idx_=dir_names.index('data')
            dir_names = dir_names[(idx_+1):]
            img_name = self.root_path
            for n in dir_names:
                img_name = os.path.join(img_name, n)


            if self.cls_list is None:
                label = self.data_frame.iloc[idx].get("label", -1)
            else:
                label = self.cls_list.index(self.data_frame.iloc[idx]["klass"])

            if self.data_dir is None:
                img_path = os.path.join("dataset", self.dataset, img_name)
            else:
                img_path = os.path.join(self.data_dir, img_name)
            image = PIL.Image.open(img_path).convert("RGB")
            if self.transform:
                image = self.transform(image)
                
            hierarchy = self.data_frame.iloc[idx]["hierarchy"]    
            cls_sup = self.data_frame.iloc[idx]["label_sup"]    
            cls_sub = self.data_frame.iloc[idx]["label_sub"]    
                
                
            sample["image"] = image
            sample["hierarchy"] = hierarchy
            sample["label"] = label
            sample["image_name"] = img_name
            sample["cls_sup"] = cls_sup
            sample["cls_sub"] = cls_sub

            if sample["hierarchy"] == 1:
                sample["label"] -= self.n_classes_sup                
                            
            return sample

    def get_image_class(self, y):
        return self.data_frame[self.data_frame["label"] == y]

    def generate_idx(self, batch_size, rand=True):
        arr = np.arange(len(self.loaded_images))
        
        if rand:
            np.random.shuffle(arr)
        
        if batch_size >= len(arr):
            return [arr]
        else:
            return np.split(arr, np.arange(batch_size, len(arr), batch_size))

    def get_data_gpu(self, indices):
        images = []
        labels = []
        hierarchy = []
        cls_sup = []
        cls_sub = []
        
        
        data = {}
        for i in indices:
            #images.append(self.transform_gpu(to_pil_image(self.loaded_images[i]["image"]))) #.to(self.device)))
            images.append(self.transform_gpu(self.loaded_images[i]["image"].to(self.device)))
            labels.append(self.loaded_images[i]["label"])
            hierarchy.append(self.loaded_images[i]["hierarchy"])
            cls_sup.append(self.loaded_images[i]["cls_sup"])
            cls_sub.append(self.loaded_images[i]["cls_sub"])
            
            
        data['image'] = torch.stack(images)
        data['label'] = torch.LongTensor(labels)            
        data['hierarchy'] = torch.LongTensor(hierarchy)            
        data['cls_sup'] = torch.LongTensor(cls_sup)            
        data['cls_sub'] = torch.LongTensor(cls_sub)            

        return data


class StreamDataset(Dataset):
    def __init__(self, root, datalist, dataset, transform, cls_list, data_dir=None, device=None, transform_on_gpu=False, \
                 n_classes_sup=None, save_test=None, test_transform=None):
        
        raw_dataset = "."
        if 'cifar100' in dataset:
            raw_data='cifar100'
        elif 'stanford_car' in dataset:
            raw_data='stanford_car'
        elif 'imagenet' in dataset:
            raw_data='imagenet'
        else:
            raise Exception('Invalid dataset')     
            
        self.root_path = f"{root}/datasets/{raw_data}/data"   
        
        self.images = []
        self.labels = []
        
        self.hierarchy = []
        self.cls_sup = []
        self.cls_sub = []
        self.n_classes_sup = n_classes_sup
        
        self.dataset = dataset
        self.transform = transform
        self.cls_list = cls_list
        self.data_dir = data_dir
        self.device = device

        self.transform_on_gpu = transform_on_gpu
        mean, std, n_classes, n_classes_sup, n_classes_sub, inp_size, _ = get_statistics(dataset=self.dataset)

        self.save_test = save_test
        if self.save_test == 'gpu':
            self.device_img = []
            self.test_transform = test_transform
            
        
        
        if self.transform_on_gpu:
            self.transform_cpu = transforms.Compose(
                [
                    transforms.Resize((inp_size, inp_size)),
                    transforms.PILToTensor()
                ])
            self.transform_gpu = transform
        for data in datalist:
            try:
                img_name = data['file_name']
            except KeyError:
                img_name = data['filepath']

            
            dir_names = img_name.split('/')
            #print(dir_names)
            idx_=dir_names.index('data')
            dir_names = dir_names[(idx_+1):]
            img_name = self.root_path
            for n in dir_names:
                img_name = os.path.join(img_name, n)
            #print(img_name)
                
            
            if self.data_dir is None:
                img_path = os.path.join("dataset", self.dataset, img_name)
            else:
                img_path = os.path.join(self.data_dir, img_name)
            #img_path = img_name
                
            img = PIL.Image.open(img_path).convert('RGB')
            if self.transform_on_gpu:
                self.images.append(self.transform_cpu(img))
            else:
                self.images.append(img)
            
            if self.save_test == 'gpu':
                self.device_img.append(self.test_transform(img))
                
            self.labels.append(self.cls_list.index(data['klass']))
            self.hierarchy.append(data['hierarchy'])
            
            if self.hierarchy[-1] == 1:
                self.labels[-1] -= self.n_classes_sup            
                #print(self.labels)
            
            
            self.cls_sup.append(data['label_sup'])
            self.cls_sub.append(data['label_sub'])
            
            

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        sample = dict()
        if torch.is_tensor(idx):
            idx = idx.tolist()
        image = self.images[idx]
        label = self.labels[idx]
        hierarchy = self.hierarchy[idx]
        
        cls_sup = self.cls_sup[idx]
        cls_sub = self.cls_sup[idx]
        
        if self.transform:
            image = self.transform(image)
        sample["image"] = image
        sample["label"] = label
        sample["hierarchy"] = hierarchy
        sample["cls_sup"] = cls_sup
        sample["cls_sub"] = cls_sub
        
        #if sample["hierarchy"] == 1:
        #    sample["label"] -= self.n_classes_sup

        
        return sample

    @torch.no_grad()
    def get_data(self):
        data = dict()
        images = []
        labels = []
        hierarchy = []
        
        cls_sup = []
        cls_sub = []
        
        for i, image in enumerate(self.images):
            if self.transform_on_gpu:
                images.append(self.transform_gpu(image.to(self.device)))
            else:
                images.append(self.transform(image))
            labels.append(self.labels[i])
            hierarchy.append(self.hierarchy[i])
            
            cls_sup.append(self.cls_sup[i])
            cls_sub.append(self.cls_sub[i])
            
            #if hierarchy[] == 1:
            #    labels[-1] -= self.n_classes_sup
        
        data['image'] = torch.stack(images)
        data['label'] = torch.LongTensor(labels)
        data['hierarchy'] = torch.LongTensor(hierarchy)
        data['cls_sup'] = torch.LongTensor(cls_sup)
        data['cls_sub'] = torch.LongTensor(cls_sub)
        
        return data
    
    @torch.no_grad()
    def get_data_for_test(self):
        data = dict()
        images = []
        labels = []
        hierarchy = []
        
        cls_sup = []
        cls_sub = []
        
        for i, image in enumerate(self.device_img):
            images.append(image.to(self.device))
            labels.append(self.labels[i])
            hierarchy.append(self.hierarchy[i])
            
            cls_sup.append(self.cls_sup[i])
            cls_sub.append(self.cls_sub[i])
            
            #if hierarchy[] == 1:
            #    labels[-1] -= self.n_classes_sup
        
        data['image'] = torch.stack(images)
        data['label'] = torch.LongTensor(labels)
        data['hierarchy'] = torch.LongTensor(hierarchy)
        data['cls_sup'] = torch.LongTensor(cls_sup)
        data['cls_sub'] = torch.LongTensor(cls_sub)
        
        return data
        
    
    
    
    
    @torch.no_grad()
    def get_two_aug_data(self, labels_only=False):
        data = dict()
        images = []
        labels = []
        hierarchy = []
        
        cls_sup = []
        cls_sub = []
        
        for i, image in enumerate(self.images):
            if not labels_only:
                if self.transform_on_gpu:
                    images.append(self.transform_gpu(image.to(self.device)))
                else:
                    images.append(self.transform(image))

                
            labels.append(self.labels[i])
            hierarchy.append(self.hierarchy[i])
            
            cls_sup.append(self.cls_sup[i])
            cls_sub.append(self.cls_sub[i])
           
        if not labels_only:
            list_1=[]
            list_2=[]
            for d in images:
                list_1.append(d[0])
                list_2.append(d[1])

            #print(d[0].shape)
            list_1 = torch.stack(list_1)
            list_2 = torch.stack(list_2)

            data['image'] = [list_1,list_2]
            
        data['label'] = torch.LongTensor(labels)
        data['hierarchy'] = torch.LongTensor(hierarchy)
        data['cls_sup'] = torch.LongTensor(cls_sup)
        data['cls_sub'] = torch.LongTensor(cls_sub)
        
        return data
    
    

class MemoryDataset(Dataset):
    def __init__(self, root, dataset, transform=None, cls_list=None, device=None, test_transform=None,
                 data_dir=None, transform_on_gpu=True, save_test=None, keep_history=False, n_classes_sup=None):
        
        raw_dataset = "."
        if 'cifar100' in dataset:
            raw_data='cifar100'
        elif 'stanford_car' in dataset:
            raw_data='stanford_car'
        elif 'imagenet' in dataset:
            raw_data='imagenet'
        else:
            raise Exception('Invalid dataset')     
            
        self.root_path = f"{root}/datasets/{raw_data}/data"
        
        
        self.datalist = []
        self.labels = []
        self.images = []
        
        
        self.hierarchy = []
        self.cls_sup = []
        self.cls_sub = []
        self.n_classes_sup = n_classes_sup
        
        
        self.dataset = dataset
        self.transform = transform
        self.cls_list = []
        self.cls_dict = {cls_list[i]:i for i in range(len(cls_list))}
        self.cls_count = []
        self.cls_idx = []
        self.cls_train_cnt = np.array([])
        self.score = []
        self.others_loss_decrease = np.array([])
        self.previous_idx = np.array([], dtype=int)
        self.device = device
        self.test_transform = test_transform
        self.data_dir = data_dir
        self.keep_history = keep_history

        self.transform_on_gpu = transform_on_gpu
        mean, std, n_classes, n_classes_sup, n_classes_sub, inp_size, _ = get_statistics(dataset=self.dataset)
        if self.transform_on_gpu:
            self.transform_cpu = transforms.Compose(
            [
                transforms.Resize((inp_size, inp_size)),
                transforms.PILToTensor()
            ])
            self.transform_gpu = transform
            self.test_transform = transforms.Compose([transforms.ConvertImageDtype(torch.float32),
            transforms.Normalize(mean, std)])
        self.save_test = save_test
        if self.save_test is not None:
            self.device_img = []
            
        
        """ For hierPL """ 
        self.len_history = 24*1024
        self.iter_encountered_cls = []
        #self.iter_encountered_hier = []
        
        self.history_trained_mem_label = []
        self.history_trained_mem_hier = []
        self.history_trained_mem_iter = []
        self.history_trained_stream_label = []
        self.history_trained_stream_hier = []
        self.history_trained_stream_iter = []
        
        self.cls_emp_prob = torch.tensor([])
        self.cls_loss_weight = []
        
        """ """         
        
        

    def __len__(self):
        return len(self.images)

    def add_new_class(self, cls_list):
        self.cls_list = cls_list
        self.cls_count.append(0)
        self.cls_idx.append([])
        self.cls_dict = {self.cls_list[i]:i for i in range(len(self.cls_list))}
        self.cls_train_cnt = np.append(self.cls_train_cnt, 0)

    def __getitem__(self, idx):
        sample = dict()
        if torch.is_tensor(idx):
            idx = idx.value()
        label = self.labels[idx]
        image = self.images[idx]
        hierarchy = self.hierarchy[idx]
        cls_sup = self.cls_sup[idx]
        cls_sub = self.cls_sub[idx]
        
        if self.transform:
            image = self.transform(image)
        sample["image"] = image
        sample["label"] = label
        sample["hierarchy"] = hierarchy
        sample["cls_sup"] = cls_sup
        sample["cls_sub"] = cls_sub
    
        if sample["hierarchy"] == 1:
            sample["label"] -= self.n_classes_sup    
        
        return sample

    def update_gss_score(self, score, idx=None):
        if idx is None:
            self.score.append(score)
        else:
            self.score[idx] = score

    def replace_sample(self, sample, idx=None):
        #input()
        
        #print(sample)
        
        self.cls_count[self.cls_dict[sample['klass']]] += 1
        if idx is None:
            self.cls_idx[self.cls_dict[sample['klass']]].append(len(self.images))
            self.datalist.append(sample)
            try:
                img_name = sample['file_name']
            except KeyError:
                img_name = sample['filepath']
            
            dir_names = img_name.split('/')
            idx_=dir_names.index('data')
            dir_names = dir_names[(idx_+1):]
            img_name = self.root_path
            for n in dir_names:
                img_name = os.path.join(img_name, n)
            
            if self.data_dir is None:
                img_path = os.path.join("dataset", self.dataset, img_name)
            else:
                img_path = os.path.join(self.data_dir, img_name)
            
            img = PIL.Image.open(img_path).convert('RGB')
            if self.transform_on_gpu:
                img = self.transform_cpu(img)
            self.images.append(img)
            self.labels.append(self.cls_dict[sample['klass']])
            self.hierarchy.append(sample['hierarchy'])
            self.cls_sup.append(sample['label_sup'])
            self.cls_sub.append(sample['label_sub'])
            
            if self.save_test == 'gpu':
                self.device_img.append(self.test_transform(img).to(self.device).unsqueeze(0))
            elif self.save_test == 'cpu':
                self.device_img.append(self.test_transform(img).unsqueeze(0))
            if self.cls_count[self.cls_dict[sample['klass']]] == 1:
                self.others_loss_decrease = np.append(self.others_loss_decrease, 0)
            else:
                self.others_loss_decrease = np.append(self.others_loss_decrease, np.mean(self.others_loss_decrease[self.cls_idx[self.cls_dict[sample['klass']]][:-1]]))
        else:
        
            self.cls_count[self.labels[idx]] -= 1
            self.cls_idx[self.labels[idx]].remove(idx)
            self.datalist[idx] = sample
            self.cls_idx[self.cls_dict[sample['klass']]].append(idx)
            try:
                img_name = sample['file_name']
            except KeyError:
                img_name = sample['filepath']
            
            dir_names = img_name.split('/')
            idx_=dir_names.index('data')
            dir_names = dir_names[(idx_+1):]
            img_name = self.root_path
            for n in dir_names:
                img_name = os.path.join(img_name, n)
            
            
            if self.data_dir is None:
                img_path = os.path.join("dataset", self.dataset, img_name)
            else:
                img_path = os.path.join(self.data_dir, img_name)
            img = PIL.Image.open(img_path).convert('RGB')
            if self.transform_on_gpu:
                img = self.transform_cpu(img)
            
            self.images[idx] = img
            self.labels[idx] = self.cls_dict[sample['klass']] #self.cls_list.index(sample['klass'])
            
            self.hierarchy[idx] = sample['hierarchy']
            self.cls_sup[idx] = sample['label_sup']
            self.cls_sub[idx] = sample['label_sub']
            
            if self.save_test == 'gpu':
                self.device_img[idx] = self.test_transform(img).to(self.device).unsqueeze(0)
            elif self.save_test == 'cpu':
                self.device_img[idx] = self.test_transform(img).unsqueeze(0)
            if self.cls_count[self.cls_dict[sample['klass']]] == 1:
                self.others_loss_decrease[idx] = np.mean(self.others_loss_decrease)
            else:
                self.others_loss_decrease[idx] = np.mean(self.others_loss_decrease[self.cls_idx[self.cls_dict[sample['klass']]][:-1]])

    def get_weight(self):
        weight = np.zeros(len(self.images))
        for i, indices in enumerate(self.cls_idx):
            weight[indices] = 1/self.cls_count[i]
        return weight

    @torch.no_grad()
    def get_batch(self, batch_size, use_weight=False, transform=None, indices=None):
        if use_weight:
            weight = self.get_weight()
            if indices is None:
                indices = np.random.choice(range(len(self.images)), size=batch_size, p=weight/np.sum(weight), replace=False)
        else:
            if indices is None:
                indices = np.random.choice(range(len(self.images)), size=batch_size, replace=False)
                
        data = dict()
        images = []
        labels = []
        hierarchy = []
        cls_sup = []
        cls_sub = []
        
        
        for i in indices:
            if transform is None:
                if self.transform_on_gpu:
                    images.append(self.transform_gpu(self.images[i].to(self.device)))
                else:
                    images.append(self.transform(self.images[i]))
            else:
                if self.transform_on_gpu:
                    images.append(transform(self.images[i].to(self.device)))
                else:
                    images.append(transform(self.images[i]))
            labels.append(self.labels[i])
            self.cls_train_cnt[self.labels[i]] += 1
            
            hierarchy.append(self.hierarchy[i])
            cls_sup.append(self.cls_sup[i])
            cls_sub.append(self.cls_sub[i])
            
            if hierarchy[-1] == 1:
                labels[-1] -= self.n_classes_sup
        
        data['image'] = torch.stack(images)
        data['label'] = torch.LongTensor(labels)
        data['hierarchy'] = torch.LongTensor(hierarchy)
        data['cls_sup'] = torch.LongTensor(cls_sup)
        data['cls_sub'] = torch.LongTensor(cls_sub)

        
        if self.keep_history:
            self.previous_idx = np.append(self.previous_idx, indices)
        return data

    
    @torch.no_grad()
    def get_two_aug_batch(self, batch_size, use_weight=False, transform=None, indices=None, labels_only=False):
        if use_weight:
            weight = self.get_weight()
            if indices is None:
                indices = np.random.choice(range(len(self.images)), size=batch_size, p=weight/np.sum(weight), replace=False)
        else:
            if indices is None:
                indices = np.random.choice(range(len(self.images)), size=batch_size, replace=False)
        
        data = dict()
        images = []
        labels = []
        hierarchy = []
        cls_sup = []
        cls_sub = []
        
        
        for i in indices:
            if not labels_only:
                if transform is None:
                    if self.transform_on_gpu:
                        images.append(self.transform_gpu(self.images[i].to(self.device)))
                    else:
                        images.append(self.transform(self.images[i]))
                else:
                    if self.transform_on_gpu:
                        images.append(transform(self.images[i].to(self.device)))
                    else:
                        images.append(transform(self.images[i]))

                    
            labels.append(self.labels[i])
            self.cls_train_cnt[self.labels[i]] += 1
            
            hierarchy.append(self.hierarchy[i])
            cls_sup.append(self.cls_sup[i])
            cls_sub.append(self.cls_sub[i])
            
            if hierarchy[-1] == 1:
                labels[-1] -= self.n_classes_sup
            
        if not labels_only:
            list_1=[]
            list_2=[]
            for d in images:
                list_1.append(d[0])
                list_2.append(d[1])

            list_1 = torch.stack(list_1)
            list_2 = torch.stack(list_2)
            data['image'] = [list_1,list_2]

        data['label'] = torch.LongTensor(labels)
        data['hierarchy'] = torch.LongTensor(hierarchy)
        data['cls_sup'] = torch.LongTensor(cls_sup)
        data['cls_sub'] = torch.LongTensor(cls_sub)

        
        if self.keep_history:
            self.previous_idx = np.append(self.previous_idx, indices)
        return data    
    
    
    def update_loss_history(self, loss, prev_loss, ema_ratio=0.5, dropped_idx=None):
        if dropped_idx is None:
            loss_diff = np.mean(loss - prev_loss)
        elif len(prev_loss) > 0:
            mask = np.ones(len(loss), bool)
            mask[dropped_idx] = False
            loss_diff = np.mean((loss[:len(prev_loss)] - prev_loss)[mask[:len(prev_loss)]])
        else:
            loss_diff = 0
        difference = loss_diff - np.mean(self.others_loss_decrease[self.previous_idx]) / len(self.previous_idx)
        self.others_loss_decrease[self.previous_idx] -= (1 - ema_ratio) * difference
        self.previous_idx = np.array([], dtype=int)

    def get_two_batches(self, batch_size, test_transform):
        indices = np.random.choice(range(len(self.images)), size=batch_size, replace=False)
        data_1 = dict()
        data_2 = dict()
        images = []
        labels = []
        hierarchy = []
        cls_sup = []
        cls_sub = []
        
        for i in indices:
            if self.transform_on_gpu:
                images.append(self.transform_gpu(self.images[i].to(self.device)))
            else:
                images.append(self.transform(self.images[i]))
            labels.append(self.labels[i])
            hierarchy.append(self.hierarchy[i])
            cls_sup.append(self.cls_sup[i])
            cls_sub.append(self.cls_sub[i])
            
            if hierarchy[-1] == 1:
                labels[-1] -= self.n_classes_sup
                        
            
            
        data_1['image'] = torch.stack(images)
        data_1['label'] = torch.LongTensor(labels)
        data_1['hierarchy'] = torch.LongTensor(hierarchy)
        data_1['cls_sup'] = torch.LongTensor(cls_sup)
        data_1['cls_sub'] = torch.LongTensor(cls_sub)
        
        images = []
        labels = []
        hierarchy = []
        cls_sup = []
        cls_sub = []
                
        for i in indices:
            images.append(self.test_transform(self.images[i]))
            labels.append(self.labels[i])
            hierarchy.append(self.hierarchy[i])
            cls_sup.append(self.cls_sup[i])
            cls_sub.append(self.cls_sub[i])
            
            if hierarchy[-1] == 1:
                labels[-1] -= self.n_classes_sup
                                    
            
        data_2['image'] = torch.stack(images)
        data_2['label'] = torch.LongTensor(labels)
        data_2['hierarchy'] = torch.LongTensor(hierarchy)
        data_2['cls_sup'] = torch.LongTensor(cls_sup)
        data_2['cls_sub'] = torch.LongTensor(cls_sub)
                
        
        return data_1, data_2

    def make_cls_dist_set(self, labels, transform=None):
        if transform is None:
            transform = self.transform
        indices = []
        for label in labels:
            indices.append(np.random.choice(self.cls_idx[label]))
        indices = np.array(indices)
        data = dict()
        images = []
        labels = []
        hierarchy = []
        cls_sup = []
        cls_sub = []
                
        for i in indices:
            images.append(transform(self.images[i]))
            labels.append(self.labels[i])
            hierarchy.append(self.hierarchy[i])
            hierarchy.append(self.hierarchy[i])
            
            if hierarchy[-1] == 1:
                labels[-1] -= self.n_classes_sup
                
            cls_sup.append(self.cls_sup[i])
            cls_sub.append(self.cls_sub[i])
            
        data['image'] = torch.stack(images)
        data['label'] = torch.LongTensor(labels)
        data['hierarchy'] = torch.LongTensor(hierarchy)
        data['cls_sup'] = torch.LongTensor(cls_sup)
        data['cls_sub'] = torch.LongTensor(cls_sub)
        
        return data

    def make_val_set(self, size=None, transform=None):
        if size is None:
            size = int(0.1*len(self.images))
        if transform is None:
            transform = self.transform
        size_per_cls = size//len(self.cls_list)
        indices = []
        for cls_list in self.cls_idx:
            if len(cls_list) >= size_per_cls:
                indices.append(np.random.choice(cls_list, size=size_per_cls, replace=False))
            else:
                indices.append(np.random.choice(cls_list, size=size_per_cls, replace=True))
        indices = np.concatenate(indices)
        data = dict()
        images = []
        labels = []
        hierarchy = []
        cls_sup = []
        cls_sub = []
        
        for i in indices:
            images.append(transform(self.images[i]))
            labels.append(self.labels[i])
            hierarchy.append(self.hierarchy[i])
            cls_sup.append(self.cls_sup[i])
            cls_sub.append(self.cls_sub[i])
            
            if hierarchy[-1] == 1:
                labels[-1] -= self.n_classes_sup            
            
            
        data['image'] = torch.stack(images)
        data['label'] = torch.LongTensor(labels)
        data['hierarchy'] = torch.LongTensor(hierarchy)
        data['cls_sup'] = torch.LongTensor(cls_sup)
        data['cls_sub'] = torch.LongTensor(cls_sub)
        
        return data

    def is_balanced(self):
        mem_per_cls = len(self.images)//len(self.cls_list)
        for cls in self.cls_count:
            if cls < mem_per_cls or cls > mem_per_cls+1:
                return False
        return True
    
    def get_stream_exclusive_indices(self, batch_size, labels, hierarchy):
        idx_sub = (hierarchy==1)
        labels += self.n_classes_sup*idx_sub.type(torch.long)
                
        excluded_labels = labels.unique()
        
        included_indices = set(range(len(self.images)))
        excluded_indices = set()
        
        for l in list(excluded_labels.numpy()):
            included_indices = included_indices - set(self.cls_idx[l])
            excluded_indices = excluded_indices.union(set(self.cls_idx[l]))
        
        included_indices = list(included_indices)
        excluded_indices = list(excluded_indices)
        
        included_subinds = np.random.choice(range(len(included_indices)), size=min(len(included_indices), batch_size), replace=False)
        indices = [included_indices[i] for i in included_subinds]
        
        if len(indices) < batch_size:
            excluded_subinds = np.random.choice(range(len(excluded_indices)), size=batch_size-len(included_indices), replace=False)
            indices += [excluded_indices[i] for i in excluded_subinds]
        
        return indices
        
        
    def update_history_trained_samples(self, memory_data, stream_data, n_count_num):
        
        labels_mem = deepcopy(memory_data['label']) + self.n_classes_sup*deepcopy(memory_data['hierarchy'])
        labels_stream = deepcopy(stream_data['label']) + self.n_classes_sup*deepcopy(stream_data['hierarchy'])
        
        
        if self.history_trained_mem_label == []:
            self.history_trained_mem_label = labels_mem
            self.history_trained_stream_label = labels_stream
            self.history_trained_iter = n_count_num*torch.ones_like(labels_mem)
            
        else:
            self.history_trained_mem_label = torch.cat([self.history_trained_mem_label, labels_mem], dim=0)
            self.history_trained_stream_label = torch.cat([self.history_trained_stream_label, labels_stream], dim=0)
            self.history_trained_iter = torch.cat([self.history_trained_iter, n_count_num*torch.ones_like(memory_data['label'])], dim=0)
            
        if self.history_trained_mem_label.shape[0] > self.len_history:
            exceed = self.history_trained_mem_label.shape[0] - self.len_history
            self.history_trained_mem_label = self.history_trained_mem_label[exceed:]
            self.history_trained_stream_label = self.history_trained_stream_label[exceed:]
            self.history_trained_iter = self.history_trained_iter[exceed:]
        
        
    def update_cls_loss_weight(self, n_count_num):
        
        labels_mem, counts_mem = torch.unique(self.history_trained_mem_label, return_counts=True)
        labels_stream, counts_stream = torch.unique(self.history_trained_stream_label, return_counts=True)
        
        self.cls_emp_prob = torch.zeros(len(self.cls_list))
        
        
        self.cls_emp_prob[labels_mem] += counts_mem.type(torch.long)
        self.cls_emp_prob[labels_stream] += counts_stream.type(torch.long)
        
        self.cls_emp_prob /= self.cls_emp_prob.sum()
        
        
        alpha = torch.clip((n_count_num-torch.tensor(self.iter_encountered_cls))/(self.len_history/2), max=1.)

        self.cls_loss_weight = alpha / (len(self.cls_list)*self.cls_emp_prob)
        
        
        
    def get_cls_loss_weight(self, y, hierarchy):
        labels = y + self.n_classes_sup*hierarchy
        return self.cls_loss_weight[labels]

    
    
    
    
    
    
    

def get_train_datalist(root, dataset, n_tasks, m, n, rnd_seed, cur_iter: int) -> List:
    if n == 100 or m == 0:
        n = 100
        m = 0
    return pd.read_json(
        f"{root}/collections/{dataset}/{dataset}_split{n_tasks}_n{n}_m{m}_rand{rnd_seed}_task{cur_iter}.json"
    ).to_dict(orient="records")

def get_test_datalist(root, dataset) -> List:
    #input(f"{root}/collections/{dataset}/{dataset}_val.json")
    
    return pd.read_json(f"{root}/collections/{dataset}/{dataset}_val.json").to_dict(orient="records")


def get_statistics(dataset: str):
    """
    Returns statistics of the dataset given a string of dataset name. To add new dataset, please add required statistics here
    """
    if dataset == 'imagenet':
        dataset = 'imagenet1000'
    assert dataset in [
        "mnist",
        "KMNIST",
        "EMNIST",
        "FashionMNIST",
        "SVHN",
        "cifar10",
        "cifar100",
        "CINIC10",
        "imagenet100",
        "imagenet1000",
        "tinyimagenet",
        "cifar100_super",
        "imagenet_subset",
        "imagenet_subset_sub_shuffle",
        "cifar100_hier_setup1",
        "cifar100_hier_setup2",
        "stanford_car_setup1",
        
        "imagenet_subset_setup2",
        "stanford_car_setup2",
    ]
    mean = {
        "mnist": (0.1307,),
        "KMNIST": (0.1307,),
        "EMNIST": (0.1307,),
        "FashionMNIST": (0.1307,),
        "SVHN": (0.4377, 0.4438, 0.4728),
        "cifar10": (0.4914, 0.4822, 0.4465),
        "cifar100": (0.5071, 0.4867, 0.4408),
        "CINIC10": (0.47889522, 0.47227842, 0.43047404),
        "tinyimagenet": (0.4802, 0.4481, 0.3975),
        "imagenet100": (0.485, 0.456, 0.406),
        "imagenet1000": (0.485, 0.456, 0.406),
        "cifar100_super": (0.5071, 0.4867, 0.4408),
        "imagenet_subset": (0.485, 0.456, 0.406),
        "imagenet_subset_sub_shuffle": (0.485, 0.456, 0.406),
        "cifar100_hier_setup1": (0.5071, 0.4867, 0.4408),
        "cifar100_hier_setup2": (0.5071, 0.4867, 0.4408),
        "stanford_car_setup1": (0.485, 0.456, 0.406),
        
        "imagenet_subset_setup2": (0.485, 0.456, 0.406),
        "stanford_car_setup2": (0.485, 0.456, 0.406),
    }

    std = {
        "mnist": (0.3081,),
        "KMNIST": (0.3081,),
        "EMNIST": (0.3081,),
        "FashionMNIST": (0.3081,),
        "SVHN": (0.1969, 0.1999, 0.1958),
        "cifar10": (0.2023, 0.1994, 0.2010),
        "cifar100": (0.2675, 0.2565, 0.2761),
        "CINIC10": (0.24205776, 0.23828046, 0.25874835),
        "tinyimagenet": (0.2302, 0.2265, 0.2262),
        "imagenet100": (0.229, 0.224, 0.225),
        "imagenet1000": (0.229, 0.224, 0.225),
        "cifar100_super": (0.2675, 0.2565, 0.2761),
        "imagenet_subset": (0.229, 0.224, 0.225),
        "imagenet_subset_sub_shuffle": (0.229, 0.224, 0.225),
        "cifar100_hier_setup1": (0.2675, 0.2565, 0.2761),
        "cifar100_hier_setup2": (0.2675, 0.2565, 0.2761),
        "stanford_car_setup1": (0.229, 0.224, 0.225),
        
        "imagenet_subset_setup2": (0.229, 0.224, 0.225),
        "stanford_car_setup2": (0.229, 0.224, 0.225),       
    }

    classes = {
        "mnist": 10,
        "KMNIST": 10,
        "EMNIST": 49,
        "FashionMNIST": 10,
        "SVHN": 10,
        "cifar10": 10,
        "cifar100": 100,
        "CINIC10": 10,
        "tinyimagenet": 200,
        "imagenet100": 100,
        "imagenet1000": 1000,
        "cifar100_super": 110,
        "imagenet_subset": 110,
        "imagenet_subset_sub_shuffle": 110,
        "cifar100_hier_setup1": 120,
        "cifar100_hier_setup2": 120,
        "stanford_car_setup1": 205,
        
        "imagenet_subset_setup2": 110,
        "stanford_car_setup2": 205,  
    }
    
    classes_sup = {
        "cifar100_super": 10,
        "imagenet_subset": 10,
        "imagenet_subset_sub_shuffle": 10,
        "cifar100_hier_setup1": 20,
        "cifar100_hier_setup2": 20,
        "stanford_car_setup1": 9,
        
        "imagenet_subset_setup2": 10,
        "stanford_car_setup2": 9,  
    }
    
    
    classes_sub = {
        "cifar100_super": 100,
        "imagenet_subset": 100,
        "imagenet_subset_sub_shuffle": 100,
        "cifar100_hier_setup1": 100,
        "cifar100_hier_setup2": 100,
        "stanford_car_setup1": 196,
        
        "imagenet_subset_setup2": 100,
        "stanford_car_setup2": 196,
    }    
    
    in_channels = {
        "mnist": 1,
        "KMNIST": 1,
        "EMNIST": 1,
        "FashionMNIST": 1,
        "SVHN": 3,
        "cifar10": 3,
        "cifar100": 3,
        "CINIC10": 3,
        "tinyimagenet": 3,
        "imagenet100": 3,
        "imagenet1000": 3,
        "cifar100_super": 3,
        "imagenet_subset": 3,
        "imagenet_subset_sub_shuffle": 3,
        "cifar100_hier_setup1": 3,
        "cifar100_hier_setup2": 3,
        "stanford_car_setup1": 3,
        
        "imagenet_subset_setup2": 3,
        "stanford_car_setup2": 3,
    }

    inp_size = {
        "mnist": 28,
        "KMNIST": 28,
        "EMNIST": 28,
        "FashionMNIST": 28,
        "SVHN": 32,
        "cifar10": 32,
        "cifar100": 32,
        "CINIC10": 32,
        "tinyimagenet": 64,
        "imagenet100": 224,
        "imagenet1000": 224,
        "cifar100_super": 32,
        "imagenet_subset": 224,
        "imagenet_subset_sub_shuffle": 224,
        "cifar100_hier_setup1": 32,
        "cifar100_hier_setup2": 32,
        "stanford_car_setup1": 224,
        
        "imagenet_subset_setup2": 224,
        "stanford_car_setup2": 224,        
    }
    return (
        mean[dataset],
        std[dataset],
        classes[dataset],
        classes_sup[dataset],
        classes_sub[dataset],
        inp_size[dataset],
        in_channels[dataset],
    )


# from https://github.com/drimpossible/GDumb/blob/74a5e814afd89b19476cd0ea4287d09a7df3c7a8/src/utils.py#L102:5
def cutmix_data(x, y, alpha=1.0, cutmix_prob=0.5):
    assert alpha > 0
    # generate mixed sample
    lam = np.random.beta(alpha, alpha)

    batch_size = x.size()[0]
    index = torch.randperm(batch_size)

    if torch.cuda.is_available():
        index = index.cuda()

    y_a, y_b = y, y[index]
    bbx1, bby1, bbx2, bby2 = rand_bbox(x.size(), lam)
    x[:, :, bbx1:bbx2, bby1:bby2] = x[index, :, bbx1:bbx2, bby1:bby2]

    # adjust lambda to exactly match pixel ratio
    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (x.size()[-1] * x.size()[-2]))
    return x, y_a, y_b, lam

""""""
def cutmix_data_pair(x_pair, y, alpha=1.0, cutmix_prob=0.5):
    assert alpha > 0
    # generate mixed sample
    lam = np.random.beta(alpha, alpha)

    x1 = x_pair[0]
    x2 = x_pair[1]
    
    batch_size = x1.size()[0]
    index = torch.randperm(batch_size)

    if torch.cuda.is_available():
        index = index.cuda()

    #print(y)
    #print(index)
    y_a, y_b = y, y[index]
    bbx1, bby1, bbx2, bby2 = rand_bbox(x1.size(), lam)
    x1[:, :, bbx1:bbx2, bby1:bby2] = x1[index, :, bbx1:bbx2, bby1:bby2]
    x2[:, :, bbx1:bbx2, bby1:bby2] = x2[index, :, bbx1:bbx2, bby1:bby2]

    # adjust lambda to exactly match pixel ratio
    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (x1.size()[-1] * x1.size()[-2]))
    return x1, x2, y_a, y_b, lam
""""""


def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1.0 - lam)
    cut_w = np.int(W * cut_rat)
    cut_h = np.int(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2




class TwoCropTransform:
    """Create two crops of the same image"""
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, x):
        return [self.transform(x), self.transform(x)]

    
class WeakStrongTransform:
    """Create two crops of the same image"""
    def __init__(self, weak_transform, strong_transform):
        self.weak_transform = weak_transform
        self.strong_transform = strong_transform

    def __call__(self, x):
        return [self.weak_transform(x), self.strong_transform(x)]
