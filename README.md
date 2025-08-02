# stack-transformer
This repository is in accompany with the paper: [Formal_Language_Learning_Transformers.pdf](Formal_Language_Learning_Transformers.pdf)
# Acknowledgement
Some parts of the code (Task setting up, regular, CS and DCF tasks) are adapted from [neural_networks_chomsky_hierarchy](https://github.com/google-deepmind/neural_networks_chomsky_hierarchy/tree/main).
## Content

```
.
├── models
|   ├── ndstack_rnn.py        - Nondeterministic Stack-RNN (DuSell & Chiang, 2021)
|   ├── rnn.py                - RNN (Elman, 1990)
|   ├── stack_rnn.py          - Stack-RNN (Joulin & Mikolov, 2015)
|   ├── tape_rnn.py           - Tape-RNN, loosely based on Baby-NTM (Suzgun et al., 2019) 
|   └── transformer.py        - Transformer (Vaswani et al., 2017)
├── tasks
|   ├── cs                    - Context-sensitive tasks
|   ├── dcf                   - Determinisitc context-free tasks
|   ├── regular               - Regular tasks
|   └── task.py               - Abstract GeneralizationTask
├── experiments
|   ├── constants.py          - Training/Evaluation constants
|   ├── curriculum.py         - Training curricula (over sequence lengths)
|   ├── example.py            - Example training script (RNN on the Even Pairs task)
|   ├── range_evaluation.py   - Evaluation loop (over unseen sequence lengths)
|   ├── training.py           - Training loop
|   └── utils.py              - Utility functions
├── README.md
├── example_stack_t.py        - Example training script
├── train_job_para.sh         - Training script submit to the remote server of Compute Canada  
└── requirements.txt          - Dependencies
```
## Dependencies
- python 3.11.2
- pytorch 2.0.1+cu118
- jaxlib 0.4.16+cuda11.cudnn86
## Setup
Install required packages:
```
pip install -r requirements.txt
```

## Deterministic Context-Free Tasks
```
cd neural_networks_chomsky_hierarchy/
python training/example.py \
    --batch_size 32 \
    --training_steps 100000 \
    --task $stack \
    --architecture transformer_encoder \
    --stack \
    --pos $pos \
    --seed 0
```
Replace `$task` with one of `["reverse_string", "stack_manipulation", "modular_arithmetic_brackets", "solve_equation"]`,
and `$pos` with one of `["NONE", "SIN_COS", "ALIBI", "RELATIVE", "ROTARY"]`.
