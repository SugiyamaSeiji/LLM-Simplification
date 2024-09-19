import random
import numpy as np
import torch
import pandas as pd
from datasets import Dataset

def fix_seed(seed):
    # random
    random.seed(seed)
    # Numpy
    np.random.seed(seed)
    # Pytorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

<<<<<<< HEAD
=======
#　指示文だけをtokenize
>>>>>>> 01115e5 (update file)
def source_prompt(data, tokenizer):
    system_prompt = "難解文を平易文に変換してください。"
    user_prompt = data["instraction"]
    input = [{"role": "system", "content": system_prompt},
             {"role": "user", "content": user_prompt},]
    encoded = tokenizer.apply_chat_template(input, tokenize=False)

    return encoded

<<<<<<< HEAD
=======
#　指示文+出力文（正解文）をtokenize
>>>>>>> 01115e5 (update file)
def all_prompt(data, tokenizer):
    system_prompt = "難解文を平易文に変換してください。"
    user_prompt = data["instraction"]
    assistant_prompt = data["output"]
    input = [{"role": "system", "content": system_prompt},
             {"role": "user", "content": user_prompt},
             {"role": "assistant", "content": assistant_prompt}]
    encoded = tokenizer.apply_chat_template(input, tokenize=False)
    return encoded

<<<<<<< HEAD
def to_dataset(comp_path, simp_path):
    comp = pd.read_csv(comp_path, names=["instraction"])
    simp = pd.read_csv(simp_path, names=["output"])
    df = pd.concat([comp, simp], axis=1)
    dataset = Dataset.from_dict(df)

    return dataset

def chat(model, instruction, tokenizer, temp=0.7, conversation=''):
    
    PROMPT_DICT = {
    "prompt_input": (
        "以下は、タスクを説明する指示と、文脈のある入力の組み合わせです。"
        "要求を適切に満たす応答を書きなさい。\n\n"
        "### 指示:\n{instruction}\n\n### 入力:{input}\n\n### 応答:"
    ),
    "prompt_no_input": (
        "以下は、タスクを説明する指示です。"
        "要求を適切に満たす応答を書きなさい。\n\n"
        "### 指示:\n{instruction}\n\n### 応答:"
    )
    }

    # 温度パラメータの入力制限
    if temp < 0.1:
        temp = 0.1
    elif temp > 1:
        temp = 1
    
    input_prompt = {
        'instruction': instruction,
        'input': conversation
    }
    
    prompt = ''
    if input_prompt['input'] == '':
        prompt = PROMPT_DICT['prompt_no_input'].format_map(input_prompt)
    else:
        prompt = PROMPT_DICT['prompt_input'].format_map(input_prompt)
    
    inputs = tokenizer(
        prompt,
        return_tensors='pt'
    ).to(model.device)
    
    with torch.no_grad():
        tokens = model.generate(
            **inputs,
            max_new_tokens=128,
            do_sample=True,
            temperature=temp,
            top_p=0.9,
            repetition_penalty=1.05,
            pad_token_id=tokenizer.pad_token_id,
        )

    output = tokenizer.decode(tokens[0], skip_special_tokens=True)
    output = output.split('\n\n')[-1].split(':')[-1]
    
    return output
=======
# データセットの読み込み
def to_dataset(comp_path, simp_path=None):
    comp = pd.read_csv(comp_path, names=["instraction"])

    if simp_path:
        simp = pd.read_csv(simp_path, names=["output"])
        df = pd.concat([comp, simp], axis=1)
        dataset = Dataset.from_dict(df)
    else:
        dataset = Dataset.from_dict(comp)

    return dataset
>>>>>>> 01115e5 (update file)
