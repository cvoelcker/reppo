#!/bin/bash
#SBATCH -N 1            # number of nodes on which to run
#SBATCH --gres=gpu:1        # number of gpus
#SBATCH --cpus-per-task=16     # number of cpus required per task
#SBATCH --mem=128GB
#SBATCH --ntasks=1
#SBATCH --tasks-per-node=1
#SBATCH --time=8:00:00      # time limit
#SBATCH --account aip-gigor
#SBATCH --job-name=mpsac_val
#SBATCH --output=slurm_logs/slurm_mjx_op_%A_%a.out
#SBATCH --error=slurm_logs/slurm_mjx_op_%A_%a.err
#SBATCH --exclude=kn104,kn115,kn146,kn153
#SBATCH --array=0-92%23

env=(AcrobotSwingup AcrobotSwingupSparse BallInCup CartpoleBalance CartpoleBalanceSparse CartpoleSwingup CartpoleSwingupSparse CheetahRun FingerSpin FingerTurnEasy FingerTurnHard FishSwim HopperHop HopperStand PendulumSwingup ReacherEasy ReacherHard WalkerRun WalkerWalk WalkerStand HumanoidStand HumanoidWalk HumanoidRun)
hostname

cd /home/$USER/projects/aip-gigor/voelcker/reppo
source .venv/bin/activate

python src/algorithms/reppo/train_reppo.py --config-name=ff_playground.yaml \
    env.name=${env[$((SLURM_ARRAY_TASK_ID%23))]} \
	seed=$RANDOM \
	+experiments=$3 \
	algorithm.actor_kl_clip_mode=$4 \
	tags=[paper_adamw,$4,$5]
