import torch_optimizer
from easydict import EasyDict as edict
from torch import optim

from models import mnist, cifar, imagenet, imagenet_subset, cifar100_super, cifar100_hier_setup1, stanford_car_setup1


def cycle(iterable):
    # iterate with shuffling
    while True:
        for i in iterable:
            yield i


def select_optimizer(opt_name, lr, model):
    if opt_name == "adam":
        params = [param for name, param in model.named_parameters() if 'fc' not in name]
        opt = optim.Adam(params, lr=lr, weight_decay=0)
        opt.add_param_group({'params': model.fc_sup.parameters()})
        opt.add_param_group({'params': model.fc_sub.parameters()})
    elif opt_name == "radam":
        params = [param for name, param in model.named_parameters() if 'fc' not in name]
        opt = torch_optimizer.RAdam(params, lr=lr, weight_decay=0.00001)
        opt.add_param_group({'params': model.fc.parameters()})
    elif opt_name == "sgd":
        params = [param for name, param in model.named_parameters() if 'fc' not in name]
        opt = optim.SGD(
            params, lr=lr, momentum=0.9, nesterov=True, weight_decay=1e-4)
        opt.add_param_group({'params': model.fc_sup.parameters()})
        opt.add_param_group({'params': model.fc_sub.parameters()})        
    else:
        raise NotImplementedError("Please select the opt_name [adam, sgd]")
    return opt

def select_scheduler(sched_name, opt, hparam=None):
    if "exp" in sched_name:
        scheduler = optim.lr_scheduler.ExponentialLR(opt, gamma=hparam)
    elif sched_name == "cos":
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            opt, T_0=1, T_mult=2
        )
    elif sched_name == "anneal":
        scheduler = optim.lr_scheduler.ExponentialLR(opt, 1 / 1.1, last_epoch=-1)
    elif sched_name == "multistep":
        scheduler = optim.lr_scheduler.MultiStepLR(
            opt, milestones=[30, 60, 80, 90], gamma=hparam
        )
    elif sched_name == "const":
        scheduler = optim.lr_scheduler.LambdaLR(opt, lambda iter: 1)
    else:
        scheduler = optim.lr_scheduler.LambdaLR(opt, lambda iter: 1)
    return scheduler


def select_model(model_name, dataset, num_classes_sup=1, num_classes_sub=1):
    opt = edict(
        {
            "depth": 18,
            "num_classes_sup": num_classes_sup,
            "num_classes_sub": num_classes_sub,
            "in_channels": 3,
            "bn": True,
            "normtype": "BatchNorm",
            "activetype": "ReLU",
            "pooltype": "MaxPool2d",
            "preact": False,
            "affine_bn": True,
            "bn_eps": 1e-6,
            "compression": 0.5,
        }
    )

    if "mnist" in dataset:
        model_class = getattr(mnist, "MLP")
    elif "cifar100_hier" in dataset:
        model_class = getattr(cifar100_hier_setup1, "ResNet")
    elif "cifar" in dataset:
        model_class = getattr(cifar, "ResNet")
    elif "imagenet_subset" in dataset:
        model_class = getattr(imagenet_subset, "ResNet")
    elif "imagenet" in dataset:
        model_class = getattr(imagenet, "ResNet")
    elif "stanford" in dataset:
        model_class = getattr(stanford_car_setup1, "ResNet")

    
    else:
        raise NotImplementedError(
            "Please select the appropriate datasets (mnist, cifar10, cifar100, imagenet)"
        )
    if model_name == "resnet18":
        opt["depth"] = 18
    elif model_name == "resnet32":
        opt["depth"] = 32
    elif model_name == "resnet34":
        opt["depth"] = 34
    elif model_name == "mlp400":
        opt["width"] = 400
    else:
        raise NotImplementedError(
            "Please choose the model name in [resnet18, resnet32, resnet34]"
        )

    model = model_class(opt)

    return model
