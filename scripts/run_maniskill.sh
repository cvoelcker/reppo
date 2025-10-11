#!/bin/bash
#SBATCH -N 1            # number of nodes on which to run
#SBATCH --gres=gpu:1        # number of gpus
#SBATCH --cpus-per-task=16     # number of cpus required per task
#SBATCH --mem=128GB
#SBATCH --ntasks=1
#SBATCH --tasks-per-node=1
#SBATCH --time=8:00:00      # time limit
#SBATCH --account aip-gigor
#SBATCH --job-name=langevin_reppo
#SBATCH --output=slurm_logs/langevin_reppo_%A_%a.out
#SBATCH --error=slurm_logs/langevin_reppo_%A_%a.err
#SBATCH --array=0-160%20

env=(PickSingleYCB-v1 UnitreeG1TransportBox-v1 PegInsertionSide-v1 UnitreeG1PlaceAppleInBowl-v1 LiftPegUpright-v1 PokeCube-v1 PullCube-v1 RollBall-v1)
hostname

cd /home/$USER/projects/aip-gigor/voelcker/reppo_maniskill/reppo
source .venv/bin/activate

uv run src/train.py --config-name=reppo_maniskill algorithm=reppo env=maniskill env.name=${env[$((SLURM_ARRAY_TASK_ID%8))]} tags=[maniskill7,policy] logging=wandb_online seed=$RANDOM +experiments=maniskill algorithm.policy_method=default algorithm.mask_truncated=False
