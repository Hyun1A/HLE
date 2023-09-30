#!/bin/bash

# CIL CONFIG
N=100
M=0
GPU_TRANSFORM="--gpu_transform"
USE_AMP="--use_amp"
NUM_GPUS=1
ROOT="../data"

EXP_NAME="exp_multi_depth"


DATASET=$1 # cifar10, cifar100, tinyimagenet, imagenet, cifar100_super, imagenet_subset, imagenet_subset_shuffle

NOTE=$2 # Short description of the experiment. (WARNING: logs/results with the same note will be overwritten!)
if [ "$NOTE" == "ewc" ]; then
    MODE="ewc++"
else
    MODE=$NOTE
fi

SEEDS=0 #"0 1 2 3 4"
GPU_IDX=0 #  0, 1, 2, 3 "1"
DEBUG=0 #"0: False, 1: True"
WORKERS_PER_GPU=4
TAU=0.75
GAMMA=0.75
LAMB=1.0


if [ "DATASET" == "cifar100"]; then
    DATASET="cifar100_hier_setup1"

elif [ "DATASET" == "stanford_car" ]; then
    DATASET="stanford_car_setup1"

elif [ "DATASET" == "imagenet" ]; then
    DATASET="imagenet_subset"
fi


if [ "$DATASET" == "cifar100_hier_setup1" ]; then
    N_TASKS=6 MEM_SIZE=2000 ONLINE_ITER=3
    MODEL_NAME="resnet34" EVAL_PERIOD=1000
    BATCHSIZE=16; LR=3e-4 OPT_NAME="adam" SCHED_NAME="default" MEMORY_EPOCH=100
    
elif [ "$DATASET" == "cifar100_hier_setup2" ]; then
    N_TASKS=6 MEM_SIZE=2000 ONLINE_ITER=3
    MODEL_NAME="resnet34" EVAL_PERIOD=1000
    BATCHSIZE=16; LR=3e-4 OPT_NAME="adam" SCHED_NAME="default" MEMORY_EPOCH=100
    
elif [ "$DATASET" == "stanford_car_setup1" ]; then
    N_TASKS=10 MEM_SIZE=1000 ONLINE_ITER=0.5
    MODEL_NAME="resnet34" EVAL_PERIOD=200
    BATCHSIZE=64; LR=3e-4 OPT_NAME="adam" SCHED_NAME="default" MEMORY_EPOCH=100
    
elif [ "$DATASET" == "stanford_car_setup2" ]; then
    N_TASKS=10 MEM_SIZE=1000 ONLINE_ITER=0.5
    MODEL_NAME="resnet34" EVAL_PERIOD=200
    BATCHSIZE=64; LR=3e-4 OPT_NAME="adam" SCHED_NAME="default" MEMORY_EPOCH=100   
    
elif [ "$DATASET" == "imagenet_subset" ]; then
    N_TASKS=6 MEM_SIZE=5000 ONLINE_ITER=0.25
    MODEL_NAME="resnet34" EVAL_PERIOD=2500
    BATCHSIZE=64; LR=3e-4 OPT_NAME="adam" SCHED_NAME="default" MEMORY_EPOCH=100   
    
elif [ "$DATASET" == "imagenet_subset_setup2" ]; then
    N_TASKS=6 MEM_SIZE=5000 ONLINE_ITER=0.25
    MODEL_NAME="resnet34" EVAL_PERIOD=2500
    BATCHSIZE=64; LR=3e-4 OPT_NAME="adam" SCHED_NAME="default" MEMORY_EPOCH=100   
    
else
    echo "Undefined setting"
    exit 1
fi

for RND_SEED in $SEEDS
do
    python main.py --mode $MODE \
    --dataset $DATASET \
    --n_tasks $N_TASKS --m $M --n $N \
    --rnd_seed $RND_SEED \
    --model_name $MODEL_NAME --opt_name $OPT_NAME --sched_name $SCHED_NAME \
    --lr $LR --batchsize $BATCHSIZE \
    --memory_size $MEM_SIZE $GPU_TRANSFORM --online_iter $ONLINE_ITER \
    --note $NOTE --eval_period $EVAL_PERIOD $USE_AMP --gpu_idx $GPU_IDX --num_gpus $NUM_GPUS --workers_per_gpu $WORKERS_PER_GPU --memory_epoch $MEMORY_EPOCH --debug $DEBUG --root $ROOT --tau $TAU --gamma $GAMMA --lamb $LAMB --exp_name $EXP_NAME
done
