# Copyright 2022 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Example script to train and evaluate a network."""

from absl import app
from absl import flags

import haiku as hk
import jax.numpy as jnp
import numpy as np
import jax
import os
import random
from absl import logging
import sys
sys.path.append('../../')
import logging
from neural_networks_chomsky_hierarchy.training import constants
from neural_networks_chomsky_hierarchy.training import curriculum as curriculum_lib
from neural_networks_chomsky_hierarchy.training import training
from neural_networks_chomsky_hierarchy.training import utils
from neural_networks_chomsky_hierarchy.models import positional_encodings as pos_encs_lib

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"]="false"
os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"]="platform"
"""Register a command‑line flag `--batch_size` (integer ≥ 1, default=128).
The Abseil flags library will parse this from sys.argv when app.run() is called.
Example usage: batch_size = _BATCH_SIZE.value
    print(f"Using batch size: {batch_size}")"""
_BATCH_SIZE = flags.DEFINE_integer(
    'batch_size',
    default=128,
    help='Training batch size.',
    lower_bound=1,
)
_TRAINING_STEPS = flags.DEFINE_integer(
    'training_steps',
    default=1_000_000,
    help='Training steps.',
    lower_bound=1,
)
# Why do we need _SEED as it seems set random later?
_SEED = flags.DEFINE_list(
    'seed',
    default=[0, 1, 2, 3, 4],
    help='Random seed.',
)
_TASK = flags.DEFINE_string(
    'task',
    default='even_pairs',
    help='Length generalization task (see `constants.py` for other tasks).',
)
_ARCHITECTURE = flags.DEFINE_string(
    'architecture',
    default='tape_rnn',
    help='Model architecture (see `constants.py` for other architectures).',
)
_POS = flags.DEFINE_string(
    'pos',
    default='NONE',
    help='Positional encodings.',
)
# Use the stacked Transformer or not.
_STACK = flags.DEFINE_boolean('stack', default=False, help='Set true to use stack')
def main(unused_argv) -> None:
  # Silence TPU backend warning from JAX
  logging.getLogger("jax._src.xla_bridge").setLevel(logging.ERROR)
  scores = []
  for seed in _SEED.value:
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    # Change your hyperparameters here. See constants.py for possible tasks and
    # architectures.

    sequence_length = 40 # Length for training
    max_range_test_length = 100 # IMPORTANT

    is_autoregressive = False
    logging.info("====================================")
    logging.info(
        "********Log info*********:Task--- %s //Training Architecture--- %s //PE--- %s //Using stack--- %s",
        _TASK.value,
        _ARCHITECTURE.value,
        _POS.value,
        _STACK.value,
    )
    if 'transformer' in _ARCHITECTURE.value:
      causal_masking = False
      if _ARCHITECTURE.value == 'transformer_decoder':
        causal_masking = True
        is_autoregressive = True
      # Parameter settings for Transformer.
      architecture_params = {'positional_encodings': pos_encs_lib.POS_ENC_TABLE[_POS.value], 'positional_encodings_params': pos_encs_lib.SinCosParams(max_time=256),
                              'stack': _STACK.value, 'feedforward': True, 'attention': True, 'num_layers': 5, 'causal_masking': causal_masking}
    elif _ARCHITECTURE.value == 'rnn':
      architecture_params = {'hidden_size': 256}
    elif _ARCHITECTURE.value == 'stack_rnn':
      architecture_params = {'hidden_size': 256, 'stack_cell_size': 8, 'stack_size': 100, 'n_stacks': 1}

    # Create the task.
    # creates a UniformCurriculum object that samples sequence from 1 to 40
    curriculum = curriculum_lib.UniformCurriculum(
        values=list(range(1, sequence_length + 1)))
    # print("Curriculum values:", curriculum._values) [1, 2, 3, 4, 5......, 40]
    if _TASK.value == 'reverse_string':
      task = constants.TASK_BUILDERS[_TASK.value](2)
    elif _TASK.value in ['solve_equation', 'modular_arithmetic_brackets', 'duplicate_string', 'modular_arithmetic', 'odds_first']:
      task = constants.TASK_BUILDERS[_TASK.value](5)
    else:
      task = constants.TASK_BUILDERS[_TASK.value]()
    # Create the model.

    computation_steps_mult = 0 # number of computation steps will match the input sequence length
    """
    task.output_length(n) returns how many tokens the model must produce for an input of length n.
    output_length(10) == 1:Does this task collapse a length‑10 input into a single output element?
    Classification tasks (e.g. “even parity?”) yield one answer, so single_output=True.
    Sequence‑to‑sequence tasks (e.g. string reversal) yield multiple outputs, so single_output=False
    """
    single_output = task.output_length(10) == 1
    if single_output:
      # Task only needs one output token, no need for autoregressive
      is_autoregressive = False

    model = constants.MODEL_BUILDERS[_ARCHITECTURE.value](
        output_size=task.output_size,
        return_all_outputs=True, # If False, the model only returns the hidden state at the last time step.
        # unpacks whatever extra hyperparameters, e.g.Transformers: num_layers, positional_encodings
        **architecture_params)
    if is_autoregressive:
      if _ARCHITECTURE.value != 'transformer':
        # At each time step, the outputs so far is given as input. Important for RNNs and LSTM
        model = utils.make_model_with_targets_as_input(
            model, computation_steps_mult
        )
      model = utils.add_sampling_to_autoregressive_model(model, single_output)
    else:
      model = utils.make_model_with_empty_targets(
          model, task, computation_steps_mult, single_output
      )
    model = hk.transform(model)

    # Create the loss and accuracy based on the pointwise ones.
    def loss_fn(output, target):
      loss = jnp.mean(jnp.sum(task.pointwise_loss_fn(output, target), axis=-1))
      return loss, {}

    def accuracy_fn(output, target):
      mask = task.accuracy_mask(target)
      return jnp.sum(mask * task.accuracy_fn(output, target)) / jnp.sum(mask)
    all_learning_rate = [1e-3, 5e-4, 3e-4, 1e-4]
    highest_accuracy = 0
    info_dict = {}
    for learning_rate in all_learning_rate:
        # Create the final training parameters.
        training_params = training.ClassicTrainingParams(
            seed=seed,
            model_init_seed=seed,
            training_steps=_TRAINING_STEPS.value,
            log_frequency=100,
            length_curriculum=curriculum,
            batch_size=_BATCH_SIZE.value,
            task=task,
            model=model,
            loss_fn=loss_fn,
            learning_rate=learning_rate,
            accuracy_fn=accuracy_fn,
            compute_full_range_test=True,
            max_range_test_length=max_range_test_length,
            range_test_total_batch_size=512,
            range_test_sub_batch_size=32,
            is_autoregressive=is_autoregressive)
        training_worker = training.TrainingWorker(training_params, use_tqdm=True)
        ## training_worker.run() return the training acc/testing acc/trained paras
        _, eval_results, _ = training_worker.run()
        accuracies = [r['accuracy'] for r in eval_results]
        score = np.mean(accuracies)
        if score > highest_accuracy:
            info_dict['learning_rate'] = learning_rate
            highest_accuracy = score
            info_dict['log'] = eval_results
        logging.info(f"Learning rate: {learning_rate}")
        logging.info(f"Average Accuracy: {score}")
        # for log in eval_results:
        #     logging.info(log)
        logging.info("------------------------------------------------------------")

    print("Best learning rate:", info_dict['learning_rate'])
    print("Best accuracy:", highest_accuracy)
    for log in info_dict['log']:
        print(log)

    # """
    # Shunqi: We do not consider the final score as we have skipped some evaluations so the sequence length is not our
    # number of test cases we used.
    # """
    # accuracies = [r['accuracy'] for r in eval_results]
    # score = np.mean(accuracies)  # ← no indexing
    #
    # for log in eval_results:    # DO the accuracy logging here
    #     print(log)
  #   print(f'seed: {seed}')
  #   print(f'Network score: {score}')
  #   scores.append(score)
  # print(f'Maximum score: {max(scores)}')
  # print(f'Mean score: {np.mean(scores)}')
  # print(f'Std score: {np.std(scores)}')

if __name__ == '__main__':
  app.run(main)
