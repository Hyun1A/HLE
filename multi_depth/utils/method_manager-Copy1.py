import logging

from methods.bic import BiasCorrection
from methods.er_baseline import ER
from methods.rainbow_memory import RM
from methods.ewc import EWCpp
from methods.mir import MIR
from methods.clib import CLIB
from methods.co2 import Co2
from methods.coreset_pseudo import CoreSetP
from methods.supcon import SupCon
from methods.con_simclr import ConSimCLR
from methods.asymsupcon import AsymSupCon
from methods.con_dclw import ConDCLW
from methods.dsupcon import DSupCon
from methods.dasymsupcon import DAsymSupCon

from methods.hp_fms import HP_FMS
from methods.hp import HP
from methods.fms import FMS
from methods.hp_sig_modules import HP_Sig_Modules

from methods.hp_fixmatch import HP_FixMatch
from methods.hp_flexmatch import HP_FlexMatch
from methods.hp_fixmatch_fms import HP_FixMatch_FMS
from methods.hp_flexmatch_fms import HP_FlexMatch_FMS
from methods.hp_openmatch import HP_OpenMatch
from methods.hp_openmatch_fms import HP_OpenMatch_FMS

from methods.hp_openmatch_oracle import HP_OpenMatch_Oracle
from methods.hp_openmatch_oracle_semi import HP_OpenMatch_Oracle_Semi
from methods.hp_openmatch_det_fixedLatent import HP_OpenMatch_Det_FixedLatent
from methods.hp_openmatch_fms_det_fixedLatent import HP_OpenMatch_FMS_Det_FixedLatent

from methods.hp_openmatch_det_fixedLatent_relaxed_node import HP_OpenMatch_Det_FixedLatent_Relaxed_Node
from methods.hp_openmatch_fms_det_fixedLatent_relaxed_node import HP_OpenMatch_FMS_Det_FixedLatent_Relaxed_Node
from methods.hp_openmatch_fms_det_fixedLatent_relaxed_node_socr import HP_OpenMatch_FMS_Det_FixedLatent_Relaxed_Node_Socr

from methods_proposed.fms_adaptive_lr import FMS_Adaptive_LR
from methods_proposed.adaptive_lr import Adaptive_LR
from methods_proposed.mo_training_adaptive_lr import MO_Training_Adaptive_LR

from methods_proposed.fms import FMS
from methods_proposed.mo_training import MO_Training
from methods_proposed.mo_loss_imp_sampling import MO_Loss_Imp_Sampling

from methods_proposed.fms_adaptive_lr_loss_imp_sampling import FMS_Adaptive_LR_Loss_Imp_Sampling
from methods_proposed.loss_imp_sampling import Loss_Imp_Sampling
from methods_proposed.loss_imp_sampling_adaptive_lr import Loss_Imp_Sampling_Adaptive_LR
from methods_proposed.fms_adaptive_lr_coreset import FMS_Adaptive_LR_Coreset
from methods_proposed.fms_adaptive_lr_entropy import FMS_Adaptive_LR_Entropy


from methods_proposed.hp_fixmatch_fms_adaptive_lr_loss_imp_sampling import HP_FixMatch_FMS_Adaptive_LR_Loss_Imp_Sampling
from methods_proposed.hp_fixmatch_fms_adaptive_lr_coreset import HP_FixMatch_FMS_Adaptive_LR_Coreset

from methods_proposed.hp_openmatch_fms_adaptive_lr_loss_imp_sampling import HP_OpenMatch_FMS_Adaptive_LR_Loss_Imp_Sampling
from methods_proposed.hp_openmatch_fms_adaptive_lr_entropy import HP_OpenMatch_FMS_Adaptive_LR_Entropy



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
    elif args.mode == "co2":
        method = Co2(
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
    elif args.mode == "coreset_pseudo":
        method = CoreSetP(
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
    elif args.mode == "supcon":
        method = SupCon(
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
    elif args.mode == "con_simclr":
        method = ConSimCLR(
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
    elif args.mode == "asymsupcon":
        method = AsymSupCon(
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
    elif args.mode == "con_dclw":
        method = ConDCLW(
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
    elif args.mode == "dsupcon":
        method = DSupCon(
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
    elif args.mode == "dasymsupcon":
        method = DAsymSupCon(
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
    elif args.mode == "hp_fms":
        method = HP_FMS(
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
    elif args.mode == "hp":
        method = HP(
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
    elif args.mode == "fms":
        method = FMS(
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
        
    elif args.mode == "hp_sig_modules":
        method = HP_Sig_Modules(
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
        
    elif args.mode == "hp_fixmatch":
        method = HP_FixMatch(
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
        
    elif args.mode == "hp_flexmatch":
        method = HP_FlexMatch(
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
        
    elif args.mode == "hp_fixmatch_fms":
        method = HP_FixMatch_FMS(
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
        
    elif args.mode == "hp_flexmatch_fms":
        method = HP_FlexMatch_FMS(
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
        
    elif args.mode == "hp_openmatch":
        method = HP_OpenMatch(
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
        
    elif args.mode == "hp_openmatch_fms":
        method = HP_OpenMatch_FMS(
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
     
        
    elif args.mode == "hp_openmatch_oracle":
        method = HP_OpenMatch_Oracle(
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
        
    elif args.mode == "hp_openmatch_oracle_semi":
        method = HP_OpenMatch_Oracle_Semi(
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
        
    elif args.mode == "hp_openmatch_det_fixedLatent":
        method = HP_OpenMatch_Det_FixedLatent(
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
        

    elif args.mode == "hp_openmatch_fms_det_fixedLatent":
        method = HP_OpenMatch_FMS_Det_FixedLatent(
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
       
    elif args.mode == "hp_openmatch_det_fixedLatent_relaxed_node":
        method = HP_OpenMatch_Det_FixedLatent_Relaxed_Node(
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
       
    elif args.mode == "hp_openmatch_fms_det_fixedLatent_relaxed_node":
        method = HP_OpenMatch_FMS_Det_FixedLatent_Relaxed_Node(
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

    elif args.mode == "hp_openmatch_fms_det_fixedLatent_relaxed_node_socr":
        method = HP_OpenMatch_FMS_Det_FixedLatent_Relaxed_Node_Socr(
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

    elif args.mode == "adaptive_lr":
        method = Adaptive_LR(
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

    elif args.mode == "fms_adaptive_lr":
        method = FMS_Adaptive_LR(
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
       
        
    elif args.mode == "mo_training_adaptive_lr":
        method = MO_Training_Adaptive_LR(
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
        
    elif args.mode == "fms":
        method = FMS(
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
        

    elif args.mode == "mo_training":
        method = MO_Training(
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
        

    elif args.mode == "mo_loss_imp_sampling":
        method = MO_Loss_Imp_Sampling(
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

    elif args.mode == "fms_adaptive_lr_loss_imp_sampling":
        method = FMS_Adaptive_LR_Loss_Imp_Sampling(
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
        
    elif args.mode == "loss_imp_sampling":
        method = Loss_Imp_Sampling(
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
        
    elif args.mode == "loss_imp_sampling_adaptive_lr":
        method = Loss_Imp_Sampling_Adaptive_LR(
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

    elif args.mode == "fms_adaptive_lr_coreset":
        method = FMS_Adaptive_LR_Coreset(
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
        
    elif args.mode == "hp_fixmatch_fms_adaptive_lr_loss_imp_sampling":
        method = HP_FixMatch_FMS_Adaptive_LR_Loss_Imp_Sampling(
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
        
    elif args.mode == "hp_fixmatch_fms_adaptive_lr_coreset":
        method = HP_FixMatch_FMS_Adaptive_LR_Coreset(
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
        
        
    elif args.mode == "fms_adaptive_lr_entropy":
        method = FMS_Adaptive_LR_Entropy(
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

        
    elif args.mode == "hp_openmatch_fms_adaptive_lr_entropy":
        method = HP_OpenMatch_FMS_Adaptive_LR_Entropy(
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




