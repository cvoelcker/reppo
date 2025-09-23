# mjx_dmc
# sbatch scripts/paper_experiments/slurm_dmc_score_ablation.sh mjx_dmc_large_data pathwise_q False  large,pathwise
# sbatch scripts/paper_experiments/slurm_dmc_score_ablation.sh mjx_dmc_large_data score_based_q False  large,score3_q,not_scaled
# sbatch scripts/paper_experiments/slurm_dmc_score_ablation.sh mjx_dmc_large_data score_based_q True  large,score3_q
sbatch scripts/paper_experiments/slurm_dmc_score_ablation.sh mjx_dmc_large_data score_based_gae False large,score4_gae
sbatch scripts/paper_experiments/slurm_dmc_score_ablation.sh mjx_dmc_large_data score_based_gae False large,score4_gae
