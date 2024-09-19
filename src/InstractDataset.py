import copy
from tqdm import tqdm
from torch.utils.data import Dataset
from utils import source_prompt, all_prompt

class InstructDataset(Dataset):
    def __init__(self, datas, tokenizer, ignore_index=-100):
        self.tokenizer = tokenizer
        self.ignore_index = ignore_index
        self.features = []
        
        for data in tqdm(datas):
<<<<<<< HEAD
            
            source_text = source_prompt(data,tokenizer) + "<|start_header_id|>assistant<|end_header_id|>"
            all_text = all_prompt(data, tokenizer)

=======
            source_text = source_prompt(data,tokenizer) + "<|start_header_id|>assistant<|end_header_id|>"
            all_text = all_prompt(data, tokenizer)


>>>>>>> 01115e5 (update file)
            # 指示文のみをtokenize
            # ほしいのは指示文のlength
            source_tokenized = self.tokenizer(
                source_text,
                padding='longest',
                truncation=True,
                max_length=512,
                return_length=True,
                return_tensors='pt'
            )
            
            # 指示文と回答文を全てtokenize
            all_tokenized = self.tokenizer(
                all_text, 
<<<<<<< HEAD
                padding='longest', 
=======
                padding=True, 
>>>>>>> 01115e5 (update file)
                truncation=True, 
                max_length=512, 
                return_tensors='pt'
            )
            
            input_ids = all_tokenized['input_ids'][0]
            
            # LLMが生成してほしい正解の文章として入力文をそのままコピーする
            labels = copy.deepcopy(input_ids)
            
            # 指示文までの長さ
            source_len = source_tokenized['length'][0]
            
            # LLMに生成してほしい正解文章に指示文も含まれているので、
            # 指示文のところはCrossEntropyLossの損失を計算をしないようにIGNORE_INDEXとして-100で埋める
            labels[:source_len] = self.ignore_index
            
            self.features.append({
                'input_ids': input_ids,
                'labels': labels
            })
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        return self.features[idx]

<<<<<<< HEAD
=======

class InstructTestDataset(Dataset):
    def __init__(self, datas, tokenizer, ignore_index=-100):
        self.tokenizer = tokenizer
        self.ignore_index = ignore_index
        self.features = []
        
        for data in tqdm(datas):
            
            source_text = source_prompt(data,tokenizer) + "<|start_header_id|>assistant<|end_header_id|>"
            source_tokenized = self.tokenizer(
                source_text,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors='pt',
            )

            input_ids = source_tokenized['input_ids'][0]
            self.features.append({
                'input_ids': input_ids,
            })
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        return self.features[idx]

>>>>>>> 01115e5 (update file)
