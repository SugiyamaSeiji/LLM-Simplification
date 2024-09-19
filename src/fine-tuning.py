<<<<<<< HEAD
import torch
import argparse
import datasets
from transformers import AutoModelForCausalLM, AutoTokenizer
from InstractDataset import InstructDataset
from InstructCollator import InstructCollator
from torch.utils.data import DataLoader
import torch.nn as nn
from peft import get_peft_model, LoraConfig, TaskType
from transformers import Trainer, TrainingArguments
from utils import fix_seed, to_dataset
import pandas as pd
from datasets import Dataset

def main():
    SEED = 42
    fix_seed(SEED)
    # parser = argparse.ArgumentParser()
    # parser.add_argument("--input_path", type=str, help="Directory where the dataset is stored.")
    # parser.add_argument("--output_path", type=str, help="Output CSV with the hypothesis.")
    # parser.add_argument("--positive_path",type=str, help="Directory where the dataset is stored.")
    # parser.add_argument("--locale", type=str, choices=['us', 'es', 'jp'], help="Locale of the queries.")
    # parser.add_argument("--model_name", type=str, help="Directory where the model is stored.")
    # parser.add_argument("--batch_size", type=int, default=256, help="Batch size.")
    # parser.add_argument("--bert_size", type=int, default=768, help="BERT size.")
    # parser.add_argument("--n_grams", type=int, help="Select gpu number")
    # args = parser.parse_args()

    model = AutoModelForCausalLM.from_pretrained("tokyotech-llm/Llama-3-Swallow-8B-Instruct-v0.1", device_map="auto", torch_dtype=torch.float16)
    tokenizer = AutoTokenizer.from_pretrained("tokyotech-llm/Llama-3-Swallow-8B-Instruct-v0.1")
    tokenizer.add_special_tokens({'unk_token': '[UNK]',
                                  'sep_token': '[SEP]',
                                  'cls_token': '[CLS]',
                                  'mask_token': '[MASK]'
                                })
    model.resize_token_embeddings(len(tokenizer))

    train_data = to_dataset("/home/sugiyama/chat-vector/data/matcha/comp.train.txt", "/home/sugiyama/chat-vector/data/matcha/simp.train.txt")
    train_dataset = InstructDataset(train_data, tokenizer)    
    collator = InstructCollator(tokenizer)
    
    for param in model.parameters():
        param.requires_grad = False # モデルをフリーズ
        if param.ndim == 1:
            # 安定のためにレイヤーノルムをfp32にキャスト
            param.data = param.data.to(torch.float32)

    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()

    class CastOutputToFloat(nn.Sequential):
        def forward(self, x): return super().forward(x).to(torch.float32)

    model.lm_head = CastOutputToFloat(model.lm_head)

    lora_config = LoraConfig(
        r=128,
        lora_alpha=128,
        target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj",],
        lora_dropout=0.05,
=======
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import torch
import torch.nn as nn
import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import EarlyStoppingCallback
from transformers import Trainer, TrainingArguments
from peft import get_peft_model, LoraConfig, TaskType
from utils import fix_seed, to_dataset
from InstractDataset import InstructDataset
from InstructCollator import InstructCollator

from trl import DataCollatorForCompletionOnlyLM

import wandb

def main():
    fix_seed(42)
    parser = argparse.ArgumentParser()
    # 訓練用データ・検証用データ
    parser.add_argument("--train_comp_file", type=str, help="train file of complicated sentences")
    parser.add_argument("--train_simp_file", type=str, help="train file of simplified sentences")
    parser.add_argument("--valid_comp_file", type=str, help="valid file of complicated sentences")
    parser.add_argument("--valid_simp_file", type=str, help="valid file of simplified sentences")
    # モデル
    parser.add_argument("--model_name", type=str, help="Directory where the model is stored.")
    parser.add_argument("--save_model", type=str, help="Directory where the model is stored.")
    # LoRaのハイパーパラメータ
    parser.add_argument("--lora_r", type=int, default=64, help="Train epochs.")
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    # 学習時のハイパーパラメータ
    parser.add_argument("--num_train_epochs", type=int, default=30, help="Train epochs.")
    parser.add_argument("--per_device_train_batch_size", type=int, default=8)
    parser.add_argument("--dataloader_num_workers", type=int, default=16)
    parser.add_argument("--lr_scheduler_type", type=str, default="constant")
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--warmup_steps", type=int, default=1000)
    parser.add_argument("--report_to", type=str, default="none")
    parser.add_argument("--fp16", type=str, default=True)
    parser.add_argument("--max_seq_length", type=int, default=256)
    # wandbのプロジェクト名
    parser.add_argument("--wandb_project", type=str, default=None)
    args = parser.parse_args()

    if args.wandb_project:
        wandb.init(project=args.wandb_project)

    model = AutoModelForCausalLM.from_pretrained(args.model_name, device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    train_data = to_dataset(args.train_comp_file, args.train_simp_file)
    valid_data = to_dataset(args.valid_comp_file, args.valid_simp_file)

    train_dataset = InstructDataset(train_data, tokenizer)
    valid_dataset = InstructDataset(valid_data, tokenizer)

    collator = InstructCollator(tokenizer)
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj",],
        lora_dropout=args.lora_dropout,
>>>>>>> 01115e5 (update file)
        bias="none",
        fan_in_fan_out=False,
        task_type=TaskType.CAUSAL_LM
    )

<<<<<<< HEAD
=======
    #model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
>>>>>>> 01115e5 (update file)
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    training_args = TrainingArguments(
<<<<<<< HEAD
        output_dir='/home/sugiyama/chat-vector/outputs',
        save_total_limit=1,
        per_device_train_batch_size=8,
        num_train_epochs=1,
        remove_unused_columns=False,
        logging_steps=20,
        fp16=True,
        dataloader_num_workers=16,
        report_to="none",
=======
        output_dir=args.save_model,
        per_device_train_batch_size=args.per_device_train_batch_size,
        num_train_epochs=args.num_train_epochs,
        dataloader_num_workers=args.dataloader_num_workers,
        lr_scheduler_type=args.lr_scheduler_type,
        learning_rate=args.learning_rate, 
        warmup_steps=args.warmup_steps,
        report_to=args.report_to,
        fp16=args.fp16,
        remove_unused_columns=False,
        logging_steps=20,
        save_total_limit=1,
        evaluation_strategy="epoch", 
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss"
>>>>>>> 01115e5 (update file)
    )

    trainer = Trainer(
            model=model,
            data_collator=collator,
            args=training_args,
            train_dataset=train_dataset,
<<<<<<< HEAD
        )

=======
            eval_dataset=valid_dataset,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
        )
    
>>>>>>> 01115e5 (update file)
    model.config.use_cache = False
    trainer.train()

if __name__ == "__main__":
    main()