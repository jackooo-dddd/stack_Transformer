#!/bin/bash
#SBATCH --time=0-00:30:00
#SBATCH --account=rrg-pfieguth
#SBATCH --mem=32000M            # memory per node
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=10      # CPU cores/threads
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

echo "Hello World"
nvidia-smi

# Activate your enviroment
source ~/envs/stack_t/bin/activate

python example_stack_t.py --batch_size 32 --training_steps 100000 --task reverse_string --architecture rnn --stack=FALSE --pos=NONE --seed=0