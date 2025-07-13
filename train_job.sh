#!/bin/bash
#SBATCH --time=0-00:30:00
#SBATCH --account=rrg-pfieguth
#SBATCH --mem=32000M            # memory per node
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=10      # CPU cores/threads
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

echo "Hello World"
nvidia-smi
module load python/3.12
module load cuda
pip uninstall -y jax jaxlib
pip install --no-index \
    jax==0.6.0 \
    jaxlib==0.6.0+computecanada

# 4) (Optional) verify inside the job
python - <<EOF
import jax
print("JAX version:", jax.__version__)
print("Devices:", jax.devices())
EOF
# Activate your enviroment
source ~/envs/stack_t/bin/activate
python ~/scratch/stack_Transformer/example_stack_t.py --batch_size 32 --training_steps 100000 --task reverse_string --architecture rnn --stack=FALSE --pos=NONE --seed=0