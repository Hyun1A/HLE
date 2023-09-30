#!/bin/bash

# CIL CONFIG
N=100
M=0
GPU_TRANSFORM="--gpu_transform"
USE_AMP="--use_amp"
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
TEMP=0.75
GAMMA=0.75
NUM_GPUS=1


if [ "DATASET" == "cifar100"]; then
    DATASET="cifar100_scene_topdown"

elif [ "DATASET" == "inat19" ]; then
    DATASET="inat19_scene_topdown"
fi

    
if [ "$DATASET" == "inat19_scene_topdown" ]; then
    N_TASKS=7 MEM_SIZE=8000 ONLINE_ITER=0.25
    MODEL_NAME="resnet34" EVAL_PERIOD=2500
    BATCHSIZE=64; LR=3e-4 OPT_NAME="adam" SCHED_NAME="default" MEMORY_EPOCH=100  
DEPTH=6

elif [ "$DATASET" == "cifar100_scene_topdown" ]; then
    N_TASKS=5 MEM_SIZE=2000 ONLINE_ITER=3
    MODEL_NAME="resnet34" EVAL_PERIOD=1000
    BATCHSIZE=16; LR=3e-4 OPT_NAME="adam" SCHED_NAME="default" MEMORY_EPOCH=100 DEPTH=4

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
    --memory_size $MEM_SIZE --gpu_transform \
    --online_iter $ONLINE_ITER \
    --note $NOTE --eval_period $EVAL_PERIOD $USE_AMP --gpu_idx $GPU_IDX --num_gpus $NUM_GPUS --workers_per_gpu $WORKERS_PER_GPU --memory_epoch $MEMORY_EPOCH --debug $DEBUG --root $ROOT --temp $TEMP --gamma $GAMMA --exp_name $EXP_NAME --depth ${DEPTH}
done
