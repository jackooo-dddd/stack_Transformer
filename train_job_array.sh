#!/bin/bash
#SBATCH --time=0-14:22:00
#SBATCH --account=def-vumaiha
#SBATCH --mem=32000M
#SBATCH --gpus-per-node=4         # Cedar only has 4 GPUs/node
#SBATCH --cpus-per-task=10
#SBATCH --output=result/PARA_%x_%A_%a.out
#SBATCH --error=result/PARA_%x_%A_%a.err
#SBATCH --array=0-10

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

# Check argument
if [[ -z "$1" ]]; then
  echo "Usage: $0 {cs|dcf|regular}"
  exit 1
fi
SUBDIR=$1
STEPS=5  # fixed number of training steps

if [[ "$SUBDIR" != "cs" && "$SUBDIR" != "dcf" && "$SUBDIR" != "regular" ]]; then
  echo "Invalid argument: $SUBDIR"
  echo "Usage: $0 {cs|dcf|regular}"
  exit 1
fi

# Load environment
module load python/3.12
module load cuda/12.6
source ~/envs/stack_t/bin/activate

# Print device info
nvidia-smi
python - <<EOF
import jax
print("JAX version:", jax.__version__)
print("Devices:", jax.devices())
EOF

# Discover tasks and select by array index
TASK_DIR=~/scratch/stack_Transformer/neural_networks_chomsky_hierarchy/tasks/$SUBDIR
# TASKS is an array holds all the jobs in the language type folder.
mapfile -t TASKS < <(find "$TASK_DIR" -name '*.py' ! -name 'task.py' -exec basename {} .py \;)
NUM=${#TASKS[@]}
if (( SLURM_ARRAY_TASK_ID >= NUM )); then
  echo "Invalid array index, But not a problem!"
  exit 1
fi
JOB_NAME=${TASKS[$SLURM_ARRAY_TASK_ID]}

echo "=== Running task ($SLURM_ARRAY_TASK_ID/$((NUM-1))): $JOB_NAME ==="

# Prepare logs
mkdir -p result/${SUBDIR}
BASE=result/${SUBDIR}/${JOB_NAME}

LOG1=${BASE}_stack_rnn.log; ERR1=${BASE}_stack_rnn.err
LOG2=${BASE}_trans_npe.log; ERR2=${BASE}_trans_npe.err
LOG3=${BASE}_trans_alibi.log; ERR3=${BASE}_trans_alibi.err
LOG4=${BASE}_stackaug_npe.log; ERR4=${BASE}_stackaug_npe.err
LOG5=${BASE}_stackaug_alibi.log; ERR5=${BASE}_stackaug_alibi.err

# timestamp function
timestamp() { date '+%Y-%m-%d %H:%M:%S'; }

# Launch 5 variants in parallel
(
  echo "1) Start STACK RNN at $(timestamp)"
  srun --exclusive -n1 --gres=gpu:1 --cpus-per-task=$SLURM_CPUS_PER_TASK \
    python ~/scratch/stack_Transformer/example_stack_t.py \
      --batch_size 32 --training_steps $STEPS \
      --task "$JOB_NAME" --architecture stack_rnn \
      --stack=False --pos=NONE --seed=0
  echo "Finish STACK RNN at $(timestamp)"
) 1> "$LOG1" 2> "$ERR1" &

(
  echo "2) Start TRANSFORMER (no PE) at $(timestamp)"
  srun --exclusive -n1 --gres=gpu:1 --cpus-per-task=$SLURM_CPUS_PER_TASK \
    python ~/scratch/stack_Transformer/example_stack_t.py \
      --batch_size 32 --training_steps $STEPS \
      --task "$JOB_NAME" --architecture transformer_encoder \
      --stack=False --pos=NONE --seed=0
  echo "Finish TRANSFORMER (no PE) at $(timestamp)"
) 1> "$LOG2" 2> "$ERR2" &

(
  echo "3) Start TRANSFORMER (ALIBI) at $(timestamp)"
  srun --exclusive -n1 --gres=gpu:1 --cpus-per-task=$SLURM_CPUS_PER_TASK \
    python ~/scratch/stack_Transformer/example_stack_t.py \
      --batch_size 32 --training_steps $STEPS \
      --task "$JOB_NAME" --architecture transformer_encoder \
      --stack=False --pos=ALIBI --seed=0
  echo "Finish TRANSFORMER (ALIBI) at $(timestamp)"
) 1> "$LOG3" 2> "$ERR3" &

(
  echo "4) Start STACK-AUG (no PE) at $(timestamp)"
  srun --exclusive -n1 --gres=gpu:1 --cpus-per-task=$SLURM_CPUS_PER_TASK \
    python ~/scratch/stack_Transformer/example_stack_t.py \
      --batch_size 32 --training_steps $STEPS \
      --task "$JOB_NAME" --architecture transformer_encoder \
      --stack=True --pos=NONE --seed=0
  echo "Finish STACK-AUG (no PE) at $(timestamp)"
) 1> "$LOG4" 2> "$ERR4" &

(
  echo "5) Start STACK-AUG (ALIBI) at $(timestamp)"
  srun --exclusive -n1 --gres=gpu:1 --cpus-per-task=$SLURM_CPUS_PER_TASK \
    python ~/scratch/stack_Transformer/example_stack_t.py \
      --batch_size 32 --training_steps $STEPS \
      --task "$JOB_NAME" --architecture transformer_encoder \
      --stack=True --pos=ALIBI --seed=0
  echo "Finish STACK-AUG (ALIBI) at $(timestamp)"
) 1> "$LOG5" 2> "$ERR5" &

# Wait for all runs\
wait
echo "=== Combined STDOUT for $JOB_NAME ==="
cat "$LOG1" "$LOG2" "$LOG3" "$LOG4" "$LOG5"

combined_err=${BASE}_all.err
  cat "$ERR1" "$ERR2" "$ERR3" "$ERR4" "$ERR5" > "$combined_err"
  echo


