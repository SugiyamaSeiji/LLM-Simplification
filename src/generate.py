import torch
import argparse
import datasets
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer
from InstractDataset import InstructTestDataset
from InstructCollator import InstructTestCollator
from utils import fix_seed, to_dataset
from peft import PeftModel

def main():
    fix_seed(42)
    parser = argparse.ArgumentParser()
    # 評価用データ
    parser.add_argument("--test_comp_file", type=str, help="test file of complicated sentences")
    # モデル
    parser.add_argument("--model_name", type=str, help="Directory where the model is stored.")
    parser.add_argument("--save_model", type=str, help="Directory where the model is stored.")
    # 推論時のハイパーパラメータ
    parser.add_argument("--per_device_test_batch_size", type=int, default=8)
    # モデル出力を保存
    parser.add_argument("--predict_file", type=str, help="predict file")
    args = parser.parse_args()

    model = AutoModelForCausalLM.from_pretrained(args.model_name, device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    tokenizer.padding_side="left"
    lora_model_path = args.save_model  # 学習後の LoRA モデルのパス
    model = PeftModel.from_pretrained(model, lora_model_path)

    test_data = to_dataset(args.test_comp_file)
    test_dataset = InstructTestDataset(test_data, tokenizer)
    collator = InstructTestCollator(tokenizer)
    batch_size = args.per_device_test_batch_size
    test_loader = DataLoader(test_dataset, batch_size=batch_size, collate_fn=collator)

    model.eval()

    generated_texts = []
    for batch in test_loader:
        input_ids = batch['input_ids'].to(model.device)
        attention_mask = batch['attention_mask'].to(model.device)
        # バッチ単位でテキスト生成
        with torch.no_grad():
            outputs = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_length=256,  
                num_beams=5,
            )
        
        for i in range(len(outputs)):
            # 指示文を削除する
            generated_tokens = outputs[i][len(input_ids[i]):]
            
            # 指示文を除き，出力結果のみをデコードする
            generated_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
            generated_texts.append(generated_text)
    
    with open(args.predict_file,"w") as f:
        for predict in generated_texts:
            predict = predict.replace("\n", "")
            f.write(f"{predict}\n")

if __name__ == "__main__":
    main()