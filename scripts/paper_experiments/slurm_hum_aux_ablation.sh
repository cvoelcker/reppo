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
#SBATCH --array=0-15%4

env=(G1JoystickFlatTerrain G1JoystickRoughTerrain T1JoystickFlatTerrain T1JoystickRoughTerrain)
hostname

cd /home/$USER/projects/aip-gigor/voelcker/reppo
source .venv/bin/activate

python python src/algorithms/reppo/train_reppo.py --config-name=ff_playground.yaml \
	env=mjx_humanoid \
    env.name=${env[$((SLURM_ARRAY_TASK_ID%4))]} \
	seed=$RANDOM \
	+experiments=$3 \
	algorithm.aux_loss_mult=0.0 \
	tags=[paper_adamw,no_aux,$4]
