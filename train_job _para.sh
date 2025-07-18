#!/bin/bash
#SBATCH --time=0-14:22:00
#SBATCH --account=def-vumaiha
#SBATCH --mem=32000M
#SBATCH --gpus-per-node=5        # request GPUs equal to parallel tasks per JOB_NAME
#SBATCH --cpus-per-task=10
#SBATCH --output=result/PARA_tasks.%j.out
#SBATCH --error=result/PARA_tasks.%j.err

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

# Check argument
if [[ -z "$1" ]]; then
  echo "Usage: $0 {cs|dcf|regular}"
  exit 1
fi
SUBDIR=$1
STEPS=10  # fixed number of training steps

if [[ "$SUBDIR" != "cs" && "$SUBDIR" != "dcf" && "$SUBDIR" != "regular" ]]; then
  echo "Invalid argument: $SUBDIR"
  echo "Usage: $0 {cs|dcf|regular}"
  exit 1
fi

module load python/3.12
module load cuda/12.6
source ~/envs/stack_t/bin/activate

nvidia-smi
python - <<EOF
import jax
print("JAX version:", jax.__version__)
print("Devices:", jax.devices())
EOF

# Ensure result directories exist
echo "Logging to result/ and result/${SUBDIR}/ with $STEPS training steps"
mkdir -p result/${SUBDIR}

TASK_DIR=~/scratch/stack_Transformer/neural_networks_chomsky_hierarchy/tasks/$SUBDIR
TASKS=$(find "$TASK_DIR" -name '*.py' ! -name 'task.py' -exec basename {} .py \;)
echo "******************* RUNNING ALL $SUBDIR TASKS *******************"

for JOB_NAME in $TASKS; do
  echo
  echo "================================================================="
  echo "=========== ITERATION FOR JOB: $JOB_NAME =========================="
  echo "================================================================="

  # Define per-variant log files
  LOG1=result/${SUBDIR}/${JOB_NAME}_stack_rnn.log
  LOG2=result/${SUBDIR}/${JOB_NAME}_trans_npe.log
  LOG3=result/${SUBDIR}/${JOB_NAME}_trans_alibi.log
  LOG4=result/${SUBDIR}/${JOB_NAME}_stackaug_npe.log
  LOG5=result/${SUBDIR}/${JOB_NAME}_stackaug_alibi.log

  # 1) STACK RNN
  (
    echo "1) Start STACK RNN for task=$JOB_NAME"
    srun --exclusive -n1 --gres=gpu:1 \
      python ~/scratch/stack_Transformer/example_stack_t.py \
        --batch_size 32 --training_steps $STEPS \
        --task "$JOB_NAME" --architecture stack_rnn \
        --stack=False --pos=NONE --seed=0
    echo "Finish STACK RNN for task=$JOB_NAME"
  ) &> "$LOG1" &

  # 2) TRANSFORMER no PE
  (
    echo "2) Start TRANSFORMER (no PE) for task=$JOB_NAME"
    srun --exclusive -n1 --gres=gpu:1 \
      python ~/scratch/stack_Transformer/example_stack_t.py \
        --batch_size 32 --training_steps $STEPS \
        --task "$JOB_NAME" --architecture transformer_encoder \
        --stack=False --pos=NONE --seed=0
    echo "Finish TRANSFORMER (no PE) for task=$JOB_NAME"
  ) &> "$LOG2" &

  # 3) TRANSFORMER with ALIBI
  (
    echo "3) Start TRANSFORMER (ALIBI) for task=$JOB_NAME"
    srun --exclusive -n1 --gres=gpu:1 \
      python ~/scratch/stack_Transformer/example_stack_t.py \
        --batch_size 32 --training_steps $STEPS \
        --task "$JOB_NAME" --architecture transformer_encoder \
        --stack=False --pos=ALIBI --seed=0
    echo "Finish TRANSFORMER (ALIBI) for task=$JOB_NAME"
  ) &> "$LOG3" &

  # 4) STACK-AUG no PE
  (
    echo "4) Start STACK-AUG (no PE) for task=$JOB_NAME"
    srun --exclusive -n1 --gres=gpu:1 \
      python ~/scratch/stack_Transformer/example_stack_t.py \
        --batch_size 32 --training_steps $STEPS \
        --task "$JOB_NAME" --architecture transformer_encoder \
        --stack=True --pos=NONE --seed=0
    echo "Finish STACK-AUG (no PE) for task=$JOB_NAME"
  ) &> "$LOG4" &

  # 5) STACK-AUG with ALIBI
  (
    echo "5) Start STACK-AUG (ALIBI) for task=$JOB_NAME"
    srun --exclusive -n1 --gres=gpu:1 \
      python ~/scratch/stack_Transformer/example_stack_t.py \
        --batch_size 32 --training_steps $STEPS \
        --task "$JOB_NAME" --architecture transformer_encoder \
        --stack=True --pos=ALIBI --seed=0
    echo "Finish STACK-AUG (ALIBI) for task=$JOB_NAME"
  ) &> "$LOG5" &

  # Wait for all 5 to complete
  wait
  echo "All variants finished for task=$JOB_NAME"

  # Concatenate logs in order back to STDOUT
  echo
  echo "------ Combined Logs for $JOB_NAME ------"
  for LOG in "$LOG1" "$LOG2" "$LOG3" "$LOG4" "$LOG5"; do
    echo "### $(basename "$LOG") ###"
    cat "$LOG"
    echo
  done

done
