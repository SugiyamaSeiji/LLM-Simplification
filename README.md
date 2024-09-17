# LLM_simplification
大規模言語モデル (LLM) を用いて難解文から平易文に変換するテキスト平易化のため訓練および推論コードです．LLMを指示チューニング (Instruction-Tuning) する上で，LoRAを使用して実装しました．
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
    └──  utils.py
