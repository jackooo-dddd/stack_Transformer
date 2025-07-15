#!/bin/bash
#SBATCH --time=0-00:20:00
#SBATCH --account=def-vumaiha
#SBATCH --mem=32000M
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=10
#SBATCH --output=result/all_tasks.%j.out
#SBATCH --error=result/all_tasks.%j.err

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

# Check argument
if [[ -z "$1" ]]; then
  echo "Usage: $0 {cs|dcf|regular}"
  exit 1
fi

SUBDIR=$1

# Validate input
if [[ "$SUBDIR" != "cs" && "$SUBDIR" != "dcf" && "$SUBDIR" != "regular" ]]; then
  echo "Invalid argument: $SUBDIR"
  echo "Usage: $0 {cs|dcf|regular}"
  exit 1
fi

# load modules & env
module load python/3.12
module load cuda/12.6
source ~/envs/stack_t/bin/activate

nvidia-smi
python - <<EOF
import jax
print("JAX version:", jax.__version__)
print("Devices:", jax.devices())
EOF

mkdir -p result

# Set task folder based on user input
TASK_DIR=~/scratch/stack_Transformer/neural_networks_chomsky_hierarchy/tasks/$SUBDIR
TASKS=$(find "$TASK_DIR" -name '*.py' ! -name 'task.py' -exec basename {} .py \;)
echo "Running All $SUBDIR Tasks"
for JOB_NAME in $TASKS; do
  echo "===========Iteration for job with name: $JOB_NAME================"
  echo "---------------------------------------------------"
  echo "Start running STACK RNN for task=$JOB_NAME"
  echo "---------------------------------------------------"
  python ~/scratch/stack_Transformer/example_stack_t.py \
      --batch_size 32 \
      --training_steps 100 \
      --task "$JOB_NAME" \
      --architecture stack_rnn \
      --stack=False \
      --pos=NONE \
      --seed=0
  echo "Finish STACK RNN for task=$JOB_NAME"

  echo "---------------------------------------------------"
  echo "Start running STANDARD transformer_encoder for task=$JOB_NAME"
  echo "---------------------------------------------------"
  python ~/scratch/stack_Transformer/example_stack_t.py \
      --batch_size 32 \
      --training_steps 100 \
      --task "$JOB_NAME" \
      --architecture transformer_encoder \
      --stack=False \
      --pos=NONE \
      --seed=0
  echo "Finish STANDARD transformer_encoder for task=$JOB_NAME"

  echo "---------------------------------------------------"
  echo "Start running STACK AUGMENTED transformer_encoder for task=$JOB_NAME"
  echo "---------------------------------------------------"
  python ~/scratch/stack_Transformer/example_stack_t.py \
      --batch_size 32 \
      --training_steps 100 \
      --task "$JOB_NAME" \
      --architecture transformer_encoder \
      --stack=True \
      --pos=NONE \
      --seed=0
  echo "Finish STACK ATTENTION transformer_encoder for task=$JOB_NAME"
done

