from torch.nn.utils.rnn import pad_sequence

class InstructCollator():
    def __init__(self, tokenizer, ignore_index=-100):
        self.tokenizer = tokenizer
        self.ignore_index = -100

    def __call__(self, examples):
        input_batch = []
        label_batch = []
        for example in examples:
            input_batch.append(example['input_ids'])
            label_batch.append(example['labels'])
        
        input_ids = pad_sequence(
            input_batch, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )

        # labelsのpaddingトークンは先程と同様にignore_indexである-100で埋める
        labels = pad_sequence(
            label_batch, batch_first=True, padding_value=self.ignore_index
        )

        # attention_maskはbool値でもいいらしい
        attention_mask = input_ids.ne(self.tokenizer.pad_token_id)
            
        return {
            'input_ids': input_ids,
            'labels': labels,
            'attention_mask': attention_mask
        }
<<<<<<< HEAD
=======

class InstructTestCollator():
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, examples):
        inputs = [example['input_ids'] for example in examples]
        
        batch = self.tokenizer.pad(
            {"input_ids": inputs},
            padding=True,
            return_tensors="pt"
        )
        
        return {
            'input_ids': batch['input_ids'],
            'attention_mask': batch['attention_mask']
        }
>>>>>>> 01115e5 (update file)
