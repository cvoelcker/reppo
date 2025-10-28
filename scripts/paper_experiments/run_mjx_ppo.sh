sbatch scripts/paper_experiments/slurm_ppo_baseline.sh ppo 256 200000000 long,small
sbatch scripts/paper_experiments/slurm_ppo_baseline.sh ppo 512 200000000 long,large

sbatch scripts/paper_experiments/slurm_ppo_baseline.sh rpo 256 200000000 long,small
sbatch scripts/paper_experiments/slurm_ppo_baseline.sh rpo 512 200000000 long,large

sbatch scripts/paper_experiments/slurm_ppo_baseline.sh dpo 256 200000000 long,small
sbatch scripts/paper_experiments/slurm_ppo_baseline.sh dpo 512 200000000 long,large
