#!/bin/bash
export CUDA_DEVICE_MAX_CONNECTIONS=1

DS_CONFIG_PATH="./config/ds_config_zero3.json"


GPUS_PER_NODE=$(python -c 'import torch; print(torch.cuda.device_count())')
NNODES=1
NODE_RANK=0
MASTER_ADDR=${MASTER_ADDR:-localhost}
MASTER_PORT=${MASTER_PORT:-6001}

DISTRIBUTED_ARGS="
    --nproc_per_node $GPUS_PER_NODE \
    --nnodes $NNODES \
    --node_rank $NODE_RANK \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT
"

torchrun $DISTRIBUTED_ARGS ../kto/kto.py \
    --deepspeed ${DS_CONFIG_PATH} \
    --per_device_train_batch_size 4 \
    --num_train_epochs 100 \
    --evaluation_strategy "no" \
    --learning_rate 1e-4 \
    --lr_scheduler_type=cosine \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 8 \
    --save_steps 100 \
    --save_total_limit 1 \
    --logging_dir ./logs_v0 \
    --logging_strategy "steps"\
    --logging_steps 10 \
    --warmup_ratio 0.1 \
    --weight_decay 0.01 \
    --adam_beta2 0.95 \
    --report_to "tensorboard" \
    --bf16 \
    --logging_first_step \
    --use_peft \
    --lora_target_modules=all-linear \
    --lora_r=16 \
    --lora_alpha=16 \
    --save_only_model \
    --output_dir "./output_dir"
