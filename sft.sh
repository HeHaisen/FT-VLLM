#!/bin/bash


python train.py \
    --model_name_or_path /hy-tmp/Qwen/ \
    --data_names "mydataset1" "mydataset2" \
    --image_folder /hy-tmp/ \
    --bf16 True \
    --freeze_backbone True \
    --output_dir ./checkpoints/model-finetune \
    --num_train_epochs 10 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --save_strategy "steps" \
    --save_steps 50000 \
    --save_total_limit 1 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 10 \
    --tf32 False \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 1 \
    --lazy_preprocess True \
    --lora_enable True \
    --report_to wandb


#deepspeed train.py \
#    --deepspeed ./zero2.json \
#    --model_name_or_path ../Qwen/ \
#    --data_path /home/hhs/kdd/test/test.json \
#    --image_folder /home/hhs/kdd/data \
#    --bf16 True \
#    --output_dir ./checkpoints/model-finetune \
#    --num_train_epochs 1 \
#    --per_device_train_batch_size 1 \
#    --per_device_eval_batch_size 1 \
#    --gradient_accumulation_steps 1 \
#    --save_strategy "steps" \
#    --save_steps 50000 \
#    --save_total_limit 1 \
#    --learning_rate 2e-5 \
#    --weight_decay 0. \
#    --warmup_ratio 0.03 \
#    --lr_scheduler_type "cosine" \
#    --logging_steps 1 \
#    --tf32 True \
#    --model_max_length 2048 \
#    --gradient_checkpointing True \
#    --dataloader_num_workers 4 \
#    --lazy_preprocess True \
#    --report_to wandb
