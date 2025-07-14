#!/bin/bash
#SBATCH --time=0-04:00:00
#SBATCH --account=def-vumaiha
#SBATCH --mem=32000M
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=10
#SBATCH --output=result/regular.%j.out
#SBATCH --error=result/regular.%j.out

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

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

# define a list of task names to run
TASKS=(
  cycle_navigation
  even_pairs
  parity_check
)

for JOB_NAME in "${TASKS[@]}"; do
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
  echo
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
  echo "Start running STACK ATTENTION transformer_encoder for task=$JOB_NAME"
  echo
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
  echo "---------------------------------------------------"
  echo
done
