#!/bin/bash
#SBATCH --time=0-00:30:00
#SBATCH --account=def-vumaiha               # <== use your actual account
#SBATCH --mem=32000M
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=10
#SBATCH --output=%x.%j.out
#SBATCH --error=%x.%j.err

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

echo "Hello World"
nvidia-smi

module load python/3.12
module load cuda/12.4  # Explicit is better than implicit

# ✅ Activate your environment FIRST
source ~/envs/stack_t/bin/activate

# ✅ Install JAX + CUDA-enabled jaxlib into your venv
pip uninstall -y jax jaxlib
pip install --no-index \
    jax==0.6.0 \
    jaxlib==0.6.0+computecanada

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
    --architecture rnn \
    --stack=FALSE \
    --pos=NONE \
    --seed=0
