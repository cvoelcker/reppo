#!/bin/bash
#SBATCH -N 1            # number of nodes on which to run
#SBATCH --gres=gpu:1        # number of gpus
#SBATCH --cpus-per-task=16     # number of cpus required per task
#SBATCH --mem=128GB
#SBATCH --ntasks=1
#SBATCH --tasks-per-node=1
#SBATCH --time=5:00:00      # time limit
#SBATCH --account aip-gigor
#SBATCH --job-name=langevin_reppo
#SBATCH --output=slurm_logs/langevin_reppo_%A_%a.out
#SBATCH --error=slurm_logs/langevin_reppo_%A_%a.err
#SBATCH --array=0-230%23

env=(AcrobotSwingup AcrobotSwingupSparse BallInCup CartpoleBalance CartpoleBalanceSparse CartpoleSwingup CartpoleSwingupSparse CheetahRun FingerSpin FingerTurnEasy FingerTurnHard FishSwim HopperHop HopperStand PendulumSwingup ReacherEasy ReacherHard WalkerRun WalkerWalk WalkerStand HumanoidStand HumanoidWalk HumanoidRun)
hostname

cd /home/$USER/projects/aip-gigor/voelcker/reppo
source .venv/bin/activate

uv run src/train.py --config-name=reppo_continuous algorithm=reppo env=mjx_dmc env.name=${env[$((SLURM_ARRAY_TASK_ID%23))]} tags=[langevin_test,single_chain,small_lr] logging=wandb_online seed=$RANDOM algorithm.num_eval=20 +experiments=mjx_dmc_large_data