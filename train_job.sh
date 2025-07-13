#!/bin/bash
#SBATCH --time=0-01:20:00
#SBATCH --account=def-vumaiha               # <== use your actual account
#SBATCH --mem=32000M
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=10
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

echo "Hello World"
nvidia-smi

module load python/3.12
module load cuda/12.6  # Explicit is better than implicit

# Activate your venv
source ~/envs/stack_t/bin/activate

## Uninstall conflicting packages
#pip uninstall -y jax jaxlib orbax-checkpoint
#
## Install the CUDA‑enabled JAX 0.4.28 wheels
#pip install --no-index \
#    jax==0.4.28 \
#    jaxlib==0.4.28+cuda12.cudnn89.computecanada

# ✅ Optional: verify GPU is visible
python - <<EOF
import jax
print("JAX version:", jax.__version__)
print("Devices:", jax.devices())
EOF

# ✅ Run your training
python ~/scratch/stack_Transformer/example_stack_t.py \
    --batch_size 32 \
    --training_steps 100000 \
    --task reverse_string \
    --architecture transformer_encoder \
    --stack=True \
    --pos=NONE \
    --seed=0

