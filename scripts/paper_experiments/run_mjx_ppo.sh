sbatch scripts/paper_experiments/slurm_ppo_baseline.sh ppo 64 200000000 long,small
sbatch scripts/paper_experiments/slurm_ppo_baseline.sh ppo 512 200000000 long,large

sbatch scripts/paper_experiments/slurm_ppo_baseline.sh rpo 64 200000000 long,small
sbatch scripts/paper_experiments/slurm_ppo_baseline.sh rpo 512 200000000 long,large

sbatch scripts/paper_experiments/slurm_ppo_baseline.sh dpo 64 200000000 long,small
sbatch scripts/paper_experiments/slurm_ppo_baseline.sh dpo 512 200000000 long,large