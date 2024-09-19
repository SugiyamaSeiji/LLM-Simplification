#!/bin/bash

TEST_COMP_FILE="../data/matcha/comp.test.txt"
MODEL_NAME="tokyotech-llm/Llama-3-Swallow-8B-Instruct-v0.1"
SAVE_MODEL="../outputs/swallow-instruct-8B/*"
PREDICT_DIR="../result/swallow-instruct-8B"
PREDICT_FILE="${PREDICT_DIR}/predict_simp.txt"
BATCH_SIZE=16

mkdir -p ${PREDICT_DIR}

# 実行コマンド
python ../src/generate.py \
  --test_comp_file $TEST_COMP_FILE \
  --model_name $MODEL_NAME \
  --save_model $SAVE_MODEL \
  --per_device_test_batch_size $BATCH_SIZE \
  --predict_file $PREDICT_FILE
