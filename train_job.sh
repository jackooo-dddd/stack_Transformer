#!/bin/bash
#SBATCH --time=0-12:22:00
#SBATCH --account=def-vumaiha
#SBATCH --mem=32000M
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=10
#SBATCH --output=result/Transformer_only.%j.out
#SBATCH --error=result/Transformer_only.%j.err

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
echo "*******************RUNNING ALL $SUBDIR TASKS*******************"
for JOB_NAME in $TASKS; do
  echo "================================================================="
  echo "===========ITERATION FOR JOB: $JOB_NAME=========================="
  echo "================================================================="
  echo "---------------------------------------------------"
  echo " 2) Start running *STANDARD* transformer_encoder for task=$JOB_NAME *WITHOUT* POSITIONAL ENCODING"
  python ~/scratch/stack_Transformer/example_stack_t.py \
      --batch_size 32 \
      --training_steps 20000 \
      --task "$JOB_NAME" \
      --architecture transformer_encoder \
      --stack=False \
      --pos=NONE \
      --seed=0
  echo "Finish running *STANDARD* transformer_encoder for task=$JOB_NAME *WITHOUT* POSITIONAL ENCODING"

  echo "---------------------------------------------------"
  echo " 3) Start running *STANDARD* transformer_encoder for task=$JOB_NAME  *WITH* POSITIONAL ENCODING"
  python ~/scratch/stack_Transformer/example_stack_t.py \
      --batch_size 32 \
      --training_steps 20000\
      --task "$JOB_NAME" \
      --architecture transformer_encoder \
      --stack=False \
      --pos=ALIBI  \
      --seed=0
  echo "Finish running *STANDARD* transformer_encoder for task=$JOB_NAME *WITH* POSITIONAL ENCODING"
done

