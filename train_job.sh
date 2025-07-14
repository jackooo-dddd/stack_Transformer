#!/bin/bash
#SBATCH --time=0-00:50:00
#SBATCH --account=def-vumaiha
#SBATCH --mem=32000M
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=10
#SBATCH --output=result/even_pairs.%j.out
#SBATCH --error=result/even_pairs.%j.out

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
JOB_NAME="even_pairs"
echo "---------------------------------------------------"
echo "Start running Stack RNN with JOB_NAME=$JOB_NAME"
python ~/scratch/stack_Transformer/example_stack_t.py \
    --batch_size 32 \
    --training_steps 1000 \
    --task "$JOB_NAME" \
    --architecture stack_rnn \
    --stack=False \
    --pos=NONE \
    --seed=0
echo "Finish running Stack RNN with JOB_NAME=$JOB_NAME"
echo "---------------------------------------------------"
echo "---------------------------------------------------"
echo "Start running STANDARD transformer_encoder with JOB_NAME=$JOB_NAME"
python ~/scratch/stack_Transformer/example_stack_t.py \
    --batch_size 32 \
    --training_steps 1000 \
    --task "$JOB_NAME" \
    --architecture transformer_encoder \
    --stack=False \
    --pos=NONE \
    --seed=0
echo "Finish running STANDARD transformer_encoder with JOB_NAME=$JOB_NAME"
echo "---------------------------------------------------"