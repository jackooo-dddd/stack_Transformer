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
|   ├── counter               - Counter tasks (Dyck, Shuffle)
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
`tasks` contains all tasks, organized in their Chomsky hierarchy levels (regular, dcf, cs, counters).
They all inherit the abstract class `GeneralizationTask`, defined in `tasks/task.py`.

`models` contains all the models we use, written in [jax](https://github.com/google/jax) and [haiku](https://github.com/deepmind/dm-haiku), two open source libraries.

`training` contains the code for training models and evaluating them on a wide range of lengths.
## Installation

Clone the source code into a local directory:
```bash
git clone [https://github.com/google-deepmind/neural_networks_chomsky_hierarchy.git](https://github.com/jackooo-dddd/stack_Transformer.git)
```

`pip install -r requirements.txt` will install all required dependencies.
This is best done inside a [conda environment](https://www.anaconda.com/).
To that end, install [Anaconda](https://www.anaconda.com/download#downloads).
Then, create and activate the conda environment:
```bash
conda create --name Stack_Attention
conda activate Stack_Attention
```

Install `pip` and use it to install all the dependencies:
```bash
conda install pip
pip install -r requirements.txt
```

If you have a GPU available (highly recommended for fast training on stack attention), then you can install JAX with CUDA support.
```bash
pip install --upgrade "jax[cuda12_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
```
Note that the jax version must correspond to the existing CUDA installation you wish to use.
Please see the [JAX documentation](https://github.com/google/jax#installation) for more details.

## Usage

Before running any code, make sure to activate the conda environment:
```bash
conda activate Stack_Attention
```
Example of Training and Running: 
```bash
python example_stack_t.py --batch_size 32 --training_steps 5000 --task shuffle2 --architecture stack_rnn --stack=True --pos=ALIBI --seed=0
```
- `$task` is the task for model to learn. Replace `$task` with one of task insider the folder 'tasks'.
- `$pos` is the positional encoding for the Transformer archetecture. Replace with one of `["NONE", "SIN_COS", "ALIBI", "RELATIVE", "ROTARY"]`.
- `$training_steps` is the total number of steps the model will spend in learning the task. The more training steps, the more data the model will learn. Adjust this parameter as appropriate.
- `$architecture`  includes Transformer and Stack Recurrent neural networks.
- `$stack` is a parameter specific to the Transformer archetecture, set it to True if you want to enable the stack attention mechnisam.
- Meanwhile, though the learning rate is not defined as a parameter, 4 learning will be used to train the model separately and only the highest accurcy will be reported, see the code 'example_stack_t.py' for more details.
