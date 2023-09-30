import logging

from methods.bic import BiasCorrection
from methods.er_baseline import ER
from methods.rainbow_memory import RM
from methods.ewc import EWCpp
from methods.mir import MIR
from methods.clib import CLIB
from methods.pl_fms import PL_FMS

logger = logging.getLogger()


def select_method(args, criterion, device, train_transform, test_transform, n_classes, n_classes_sup, n_classes_sub, writer):
    kwargs = vars(args)
    if args.mode == "er":
        method = ER(
            criterion=criterion,
            device=device,
            train_transform=train_transform,
            test_transform=test_transform,
            n_classes=n_classes,
            n_classes_sup=n_classes_sup,
            n_classes_sub=n_classes_sub,
            writer=writer,
            **kwargs,
        )
    elif args.mode == "gdumb":
        from methods.gdumb import GDumb
        method = GDumb(
            criterion=criterion,
            device=device,
            train_transform=train_transform,
            test_transform=test_transform,
            n_classes=n_classes,
            n_classes_sup=n_classes_sup,
            n_classes_sub=n_classes_sub,
            writer=writer,
            **kwargs,
        )
    elif args.mode == "rm":        
        method = RM(
            criterion=criterion,
            device=device,
            train_transform=train_transform,
            test_transform=test_transform,
            n_classes=n_classes,
            n_classes_sup=n_classes_sup,
            n_classes_sub=n_classes_sub,
            writer=writer,
            **kwargs,
        )
        
    elif args.mode == "bic":
        method = BiasCorrection(
            criterion=criterion,
            device=device,
            train_transform=train_transform,
            test_transform=test_transform,
            n_classes=n_classes,
            n_classes_sup=n_classes_sup,
            n_classes_sub=n_classes_sub,
            writer=writer,
            **kwargs,
        )
    elif args.mode == "ewc++":
        method = EWCpp(
            criterion=criterion,
            device=device,
            train_transform=train_transform,
            test_transform=test_transform,
            n_classes=n_classes,
            n_classes_sup=n_classes_sup,
            n_classes_sub=n_classes_sub,
            writer=writer,
            **kwargs,
        )
    elif args.mode == "mir":
        method = MIR(
            criterion=criterion,
            device=device,
            train_transform=train_transform,
            test_transform=test_transform,
            n_classes=n_classes,
            n_classes_sup=n_classes_sup,
            n_classes_sub=n_classes_sub,
            writer=writer,
            **kwargs,
        )
    elif args.mode == "clib":
        method = CLIB(
            criterion=criterion,
            device=device,
            train_transform=train_transform,
            test_transform=test_transform,
            n_classes=n_classes,
            n_classes_sup=n_classes_sup,
            n_classes_sub=n_classes_sub,
            writer=writer,
            **kwargs,
        )
            
    elif args.mode == "pl_fms":
        method = PL_FMS(
            criterion=criterion,
            device=device,
            train_transform=train_transform,
            test_transform=test_transform,
            n_classes=n_classes,
            n_classes_sup=n_classes_sup,
            n_classes_sub=n_classes_sub,
            writer=writer,
            **kwargs,
        )    
                
        
        
    else:
        raise NotImplementedError("Choose the args.mode in [er, gdumb, rm, bic, ewc++, mir, clib]")

    return method




