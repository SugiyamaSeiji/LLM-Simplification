#!/bin/bash

#!/bin/bash

# 設定
TRAIN_COMP_FILE="../data/matcha/comp.train.txt"
TRAIN_SIMP_FILE="../data/matcha/simp.train.txt"
VALID_COMP_FILE="../data/matcha/comp.val.txt"
VALID_SIMP_FILE="../data/matcha/simp.val.txt"
MODEL_NAME="tokyotech-llm/Llama-3-Swallow-8B-Instruct-v0.1"
SAVE_MODEL_DIR="../outputs/swallow-instruct-8B"
LORA_R=64
LORA_ALPHA=16
LORA_DROPOUT=0.05
NUM_TRAIN_EPOCHS=1
PER_DEVICE_TRAIN_BATCH_SIZE=8
DATALOADER_NUM_WORKERS=16
LR_SCHEDULER_TYPE="constant"
LEARNING_RATE=2e-4
WARMUP_STEPS=1000
REPORT_TO="wandb"
FP16=True
MAX_SEQ_LENGTH=256
WANDB_PROJECT="swallow-instruct-8B"

# Pythonスクリプトの実行
python ../src/fine-tuning.py \
    --train_comp_file $TRAIN_COMP_FILE \
    --train_simp_file $TRAIN_SIMP_FILE \
    --valid_comp_file $VALID_COMP_FILE \
    --valid_simp_file $VALID_SIMP_FILE \
    --model_name $MODEL_NAME \
    --save_model $SAVE_MODEL_DIR \
    --lora_r $LORA_R \
    --lora_alpha $LORA_ALPHA \
    --lora_dropout $LORA_DROPOUT \
    --num_train_epochs $NUM_TRAIN_EPOCHS \
    --per_device_train_batch_size $PER_DEVICE_TRAIN_BATCH_SIZE \
    --dataloader_num_workers $DATALOADER_NUM_WORKERS \
    --lr_scheduler_type $LR_SCHEDULER_TYPE \
    --learning_rate $LEARNING_RATE \
    --warmup_steps $WARMUP_STEPS \
    --report_to $REPORT_TO \
    --fp16 $FP16 \
    --max_seq_length $MAX_SEQ_LENGTH \
    --wandb_project $WANDB_PROJECT

