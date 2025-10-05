#!/bin/bash
#SBATCH -N 1            # number of nodes on which to run
#SBATCH --gres=gpu:1        # number of gpus
#SBATCH --cpus-per-task=16     # number of cpus required per task
#SBATCH --mem=64GB
#SBATCH --ntasks=1
#SBATCH --tasks-per-node=1
#SBATCH --time=3:00:00      # time limit
#SBATCH --account aip-gigor
#SBATCH --job-name=ppo_val
#SBATCH --output=slurm_logs/slurm_mjx_op_%A_%a.out
#SBATCH --error=slurm_logs/slurm_mjx_op_%A_%a.err
#SBATCH --exclude=kn159
#SBATCH --array=0-460%50

env=(AcrobotSwingup AcrobotSwingupSparse BallInCup CartpoleBalance CartpoleBalanceSparse CartpoleSwingup CartpoleSwingupSparse CheetahRun FingerSpin FingerTurnEasy FingerTurnHard FishSwim HopperHop HopperStand PendulumSwingup ReacherEasy ReacherHard WalkerRun WalkerWalk WalkerStand HumanoidStand HumanoidWalk HumanoidRun)

hostname

cd /home/$USER/projects/aip-gigor/voelcker/reppo
source .venv/bin/activate

uv run src/train.py --config-name=ppo_continuous \
    algorithm=$1 \
	env=mjx_dmc \
    env.name=${env[$((SLURM_ARRAY_TASK_ID%23))]} \
	tags=[paper_adamw,ppo_paper2,tanh,$1,$4] \
	logging=wandb_online \
	algorithm.total_time_steps=$3 \
	algorithm.hidden_dim=$2 \
	algorithm.num_eval=100 \
	algorithm.use_tanh_gaussian=True \
	seed=$RANDOM

