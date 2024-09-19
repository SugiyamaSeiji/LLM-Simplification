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
from utils import fix_seed, chat
from peft import PeftModel

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
    dolly_ja = datasets.load_dataset("kunishou/databricks-dolly-15k-ja")
    dolly_ja = list(dolly_ja['train'])

    train_dataset = InstructDataset(dolly_ja, tokenizer)
    collator = InstructCollator(tokenizer)
    
    lora_model_path = "/home/sugiyama/chat-vector/outputs/checkpoint-1500"  # 学習後の LoRA モデルのパス
    model = PeftModel.from_pretrained(model, lora_model_path)

    print(chat(model, "日本の観光名所を3つ挙げて．", tokenizer, 0.7))

if __name__ == "__main__":
    main()