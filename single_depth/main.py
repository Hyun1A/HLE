from time import time
from time import localtime

import logging.config
import os, sys
import random
import pickle
from collections import defaultdict
from copy import deepcopy

import numpy as np
import torch
from randaugment import RandAugment
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

from configuration import config
from utils.augment import Cutout, select_autoaugment
from utils.data_loader import get_test_datalist, get_statistics
from utils.data_loader import get_train_datalist
from utils.method_manager import select_method

tm = time()
loc_tm = localtime(tm)


def main():
    args = config.base_parser()

    args.loc_tm = loc_tm
    args.tm = f'{args.loc_tm.tm_year}_{args.loc_tm.tm_mon}_{args.loc_tm.tm_mday}_{args.loc_tm.tm_hour}_{args.loc_tm.tm_min}_{args.loc_tm.tm_sec}'
    
    res_path =  f"{args.root}/results/{args.dataset}/{args.mode}/memory_size_{args.memory_size}/seed_{args.rnd_seed}/{args.tm}/"
    res_path_latest =  f"{args.root}/results/{args.dataset}/{args.mode}/memory_size_{args.memory_size}/seed_{args.rnd_seed}/latest/"
    
    tensorboard_path = f"{args.root}/tensorboard/{args.dataset}/{args.mode}/memory_size_{args.memory_size}/{args.rnd_seed}/{args.tm}/"
    
    os.makedirs(res_path, exist_ok=True)

    
    if args.debug == 0:
        args.debug = False
    else:
        args.debug = True

    """
    if args.mode == 'gdumb':
        if args.num_gpus == 1:
            os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        elif args.num_gpus == 2:
            os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
        elif args.num_gpus == 2:
            os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2"
        elif args.num_gpus == 2:
            os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"            
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_idx
    """
    
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_idx

    logging.config.fileConfig("./configuration/logging.conf")
    logger = logging.getLogger()

    #os.makedirs(f"results/{args.dataset}/{args.note}", exist_ok=True)
    os.makedirs(tensorboard_path, exist_ok=True)
    fileHandler = logging.FileHandler(os.path.join(res_path, 'train_log.log'), mode="w")

    formatter = logging.Formatter(
        "[%(levelname)s] %(filename)s:%(lineno)d > %(message)s"
    )
    fileHandler.setFormatter(formatter)
    logger.addHandler(fileHandler)

    writer = SummaryWriter(tensorboard_path)

    logger.info(args)

    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
        if args.gpu_transform:
            args.gpu_transform = False
            logger.warning("Augmentation on GPU not available!")
    logger.info(f"Set the device ({device})")

    # Fix the random seeds
    torch.manual_seed(args.rnd_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(args.rnd_seed)
    random.seed(args.rnd_seed)

    # Transform Definition
    mean, std, n_classes, n_classes_sup, n_classes_sub, inp_size, _ = get_statistics(dataset=args.dataset)
    train_transform = []
    if "cutout" in args.transforms:
        train_transform.append(Cutout(size=16))
        if args.gpu_transform:
            args.gpu_transform = False
            logger.warning("cutout not supported on GPU!")
    if "randaug" in args.transforms:
        train_transform.append(RandAugment())
        if args.gpu_transform:
            args.gpu_transform = False
            logger.warning("randaug not supported on GPU!")
    if "autoaug" in args.transforms:
        if 'cifar' in args.dataset:
            train_transform.append(transforms.AutoAugment(transforms.AutoAugmentPolicy('cifar10')))
        else:
            train_transform.append(transforms.AutoAugment(transforms.AutoAugmentPolicy('imagenet')))
    
    if args.gpu_transform:
        train_transform = transforms.Compose([
            transforms.RandomCrop(inp_size, padding=4),
            transforms.RandomHorizontalFlip(),
            *train_transform,
            transforms.ConvertImageDtype(torch.float32),
            transforms.Normalize(mean, std),
        ])
        
        
    else:
        train_transform = transforms.Compose(
            [
                transforms.Resize((inp_size, inp_size)),
                transforms.RandomCrop(inp_size, padding=4),
                transforms.RandomHorizontalFlip(),
                *train_transform,
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ]
        )
        
    if "hp" in args.mode:
        from utils.data_loader import WeakStrongTransform
        weak_transform = transforms.Compose(
            [
                transforms.RandomCrop(inp_size, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ConvertImageDtype(torch.float32),
                transforms.Normalize(mean, std),
            ]
        )
        strong_transform = train_transform
        
        train_transform = WeakStrongTransform(weak_transform, strong_transform)
    
    logger.info(f"Using train-transforms {train_transform}")

    test_transform = transforms.Compose(
        [
            transforms.Resize((inp_size, inp_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )

    logger.info(f"[1] Select a CIL method ({args.mode})")
    criterion = nn.CrossEntropyLoss(reduction="sum")
        
    method = select_method(
        args, criterion, device, train_transform, test_transform, n_classes, n_classes_sup, n_classes_sub, writer
    )    
    
    logger.info(f"[2] Incrementally training {args.n_tasks} tasks")
    
    task_records_sup = defaultdict(list)
    eval_results_sup = defaultdict(list)
    
    task_records_sub = defaultdict(list)
    eval_results_sub = defaultdict(list)
    
    task_records_sup_only = defaultdict(list)
    eval_results_sup_only = defaultdict(list)
    
    samples_cnt = 0
    test_datalist = get_test_datalist(args.root, args.dataset)
    test_datalist_debug = deepcopy(test_datalist)

   # if args.debug:
   #     test_datalist = test_datalist[:1000] + test_datalist[5000:5200] + test_datalist[6000:6200] + test_datalist[7000:7200] + test_datalist[8000:8200] + test_datalist[9000:9200]

    
    for cur_iter in range(args.n_tasks):
        method.cur_iter = cur_iter
        if args.mode == 'coreset_pseudo':
            if cur_iter > 0:
                method.sample_mode = 'super_first'
            else:
                method.sample_mode = 'balanced'
        
        if args.mode == "joint" and cur_iter > 0:
            return

        print("\n" + "#" * 50)
        print(f"# Task {cur_iter} iteration")
        print("#" * 50 + "\n")
        logger.info("[2-1] Prepare a datalist for the current task")

        # get datalist
        cur_train_datalist = get_train_datalist(args.root, args.dataset, args.n_tasks, args.m, args.n, args.rnd_seed, cur_iter)        
        
        
        # Reduce datalist in Debug mode
        print('debug:', args.debug)
        if args.debug:
            cur_train_datalist = cur_train_datalist[:201]
            random.shuffle(test_datalist_debug)
            test_datalist = test_datalist_debug[:201]# + test_datalist_debug[-150:]
            args.eval_period = 75
            method.eval_period = args.eval_period
            args.memory_epoch = 10
            method.memory_size = 300
            method.memory_epoch = args.memory_epoch
                
        method.online_before_task(cur_iter)
        for i, data in enumerate(cur_train_datalist):
            #print('data', data)
            
            samples_cnt += 1
            if ('hp' in args.mode) and ('entropy' in args.mode):
                method.online_step(data, samples_cnt, args.n_worker, strong_transform)
            else:
                method.online_step(data, samples_cnt, args.n_worker)
                
            if samples_cnt % args.eval_period == 0:
                #if args.mode == 'co2':
                #    method.online_after_task()
                
                
                eval_dict_sup, eval_dict_sub, eval_dict_sup_only = method.online_evaluate(test_datalist, samples_cnt, 512, args.n_worker)
                
                eval_results_sup = append_eval_results(eval_results_sup, eval_dict_sup, samples_cnt)
                eval_results_sub = append_eval_results(eval_results_sub, eval_dict_sub, samples_cnt)
                eval_results_sup_only = append_eval_results(eval_results_sup_only, eval_dict_sup_only, samples_cnt)
            
                
        method.online_after_task(cur_iter)
        
        
        eval_dict_sup, eval_dict_sub, eval_dict_sup_only = method.online_evaluate(test_datalist, samples_cnt, 512, args.n_worker, end_task=True)
        
        
        if args.mode != 'gdumb':
            task_records_sup = append_task_records(task_records_sup, eval_dict_sup, cur_iter, logger, writer)
            task_records_sub = append_task_records(task_records_sub, eval_dict_sub, cur_iter, logger, writer)
            task_records_sup_only = append_task_records(task_records_sup_only, eval_dict_sup_only, cur_iter, logger, writer)

    #if args.mode != 'gdumb':
    #    np.save(os.path.join(res_path, "sup.npy"), task_records_sup["task_acc"])
    #    np.save(os.path.join(res_path, "sub.npy"), task_records_sub["task_acc"])
    #    np.save(os.path.join(res_path, "sup_only.npy"), task_records_sup_only["task_acc"])

    
    if args.mode == 'gdumb':
        print(args.mode)
        print('memory epoch:', args.memory_epoch)
        print('batch size:', args.batchsize)
        print('n_worker:', args.n_worker)
        
        method.evaluate_all(test_datalist, args.memory_epoch, args.batchsize, args.n_worker)
                # evaluate_all: do train and evaluation simultaneously
    #else:
    #    print()
    #    summary(args, eval_results_sup, task_records_sup, n_classes_sup, 'superclasses', logger, writer)
    #    print()
    #    summary(args, eval_results_sub, task_records_sub, n_classes_sub, 'subclasses', logger, writer)
    #    print()
    #    summary(args, eval_results_sup_only, task_records_sup_only, n_classes_sup, 'superclasses only', logger, writer)   

    writer.close()

    
def append_eval_results(eval_results, eval_dict, samples_cnt):
    eval_results["test_acc"].append(eval_dict['avg_acc'])
    eval_results["avg_acc"].append(eval_dict['cls_acc'])
    eval_results["data_cnt"].append(samples_cnt)

    return eval_results

    
def append_task_records(task_records, eval_dict, cur_iter, logger, writer):
    task_acc = eval_dict['avg_acc']

    logger.info("[2-4] Update the information for the current task")
    task_records["task_acc"].append(task_acc)
    task_records["cls_acc"].append(eval_dict["cls_acc"])

    logger.info("[2-5] Report task result")
    writer.add_scalar("Metrics/TaskAcc", task_acc, cur_iter)

    return task_records


def summary(args, eval_results, task_records, n_classes, hierarchy, logger, writer):
    print('summary of results for '+ hierarchy)

    #if args.eval_period is not None:
    #    np.save(os.path.join(res_path, 'eval.npy'), eval_results['test_acc'])
    #    np.save(os.path.join(res_path, 'eval_time.npy'), eval_results['data_cnt'])

    # Accuracy (A)
    A_auc = np.mean(eval_results["test_acc"])
    A_avg = np.mean(task_records["task_acc"])
    A_last = task_records["task_acc"][args.n_tasks - 1]

    # Forgetting (F)
    cls_acc = np.array(task_records["cls_acc"])
    acc_diff = []
    for j in range(n_classes):
        if np.max(cls_acc[:-1, j]) > 0:
            acc_diff.append(np.max(cls_acc[:-1, j]) - cls_acc[-1, j])
    F_last = np.mean(acc_diff)

    logger.info(f"======== Summary =======")
    logger.info(f"A_auc {A_auc} | A_avg {A_avg} | A_last {A_last} | F_last {F_last}")    
    
    
    
if __name__ == "__main__":
    main()
