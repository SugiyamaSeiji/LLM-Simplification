# LoRAを用いたInstruction-Tuning

大規模言語モデル (LLM) を用いて難解文から平易文に変換するテキスト平易化のため訓練および推論コードです．
テキスト平易化にはMATCHAコーパスを使用し，LLMを指示チューニング (Instruction-Tuning) する上でLoRAを使用して実装しました．
以下は，フォルダ/ファイル構成になっています．
<pre>
├── data
│   └── matcha
│       ├── comp.test.txt
│       ├── comp.train.txt
│       ├── comp.val.txt
│       ├── simp.train.txt
│       └── simp.val.txt
├── scripts
│   ├── fine-tuning.sh
│   └──  inference.sh
│ 
└── src
    ├── InstractDataset.py
    ├── InstructCollator.py
    ├── fine-tuning.py
    ├── generate.py
    └── utils.py
</pre>


InstractDataset.py
- 変数source_textが指示文（<|begin_of_text|><|start_header_id|>system<|end_header_id|>...<|start_header_id|>assistant<|end_header_id|>）までの部分を示しています
- 変数all_textが指示文も含めた全テキスト部分を示しています
- alltextのうち先頭からsource_textの長さ分だけ-100で埋めることで，指示文の損失を計算しないようにしています

InstructCollator.py
- labels（正解文：平易文）においては，指示文のところを-100でpaddingしています
- 最後にinput_idsとlabels，attention_maskを戻すようにしています

# 訓練
今回使用したLLMは，[tokyotech-llm/Llama-3-Swallow-8B-Instruct-v0.1](https://huggingface.co/tokyotech-llm/Llama-3-Swallow-8B-Instruct-v0.1)です．
```
cd scripts
./fine-tuning.sh
```

# 推論
評価用データを使用して推論します．
推論した結果は，resultフォルダに保存されます．
```
cd scripts
./inference.sh
```

# 引用
訓練および推論コードの作成にあたり以下を参考にしています．
- https://qiita.com/m__k/items/173ade78990b7d6a4be4
- https://github.com/ghmagazine/llm-book  
