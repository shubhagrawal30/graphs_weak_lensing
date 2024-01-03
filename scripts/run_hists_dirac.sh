#!/bin/bash 
#SBATCH -A des 
#SBATCH -C cpu 
#SBATCH -q regular 
#SBATCH -t 10:00:00 
#SBATCH --nodes=1
#SBATCH --cpus-per-task=128
#SBATCH --job-name=run_hists_dirac
#SBATCH --mem=0
#SBATCH --mail-type=ALL
#SBATCH --mail-user=shubh@sas.upenn.edu

module load python
conda activate gnn
cd /global/cfs/cdirs/des/shubh/graphs/graphs_weak_lensing/scripts
# srun --cpu-bind=none python -c "from hist_dirac import DiracHistograms; DiracHistograms('20231115dirac_tomobin0_scale21.0')"
# srun --cpu-bind=none python -c "from hist_dirac import DiracHistograms; DiracHistograms('20231128dirac', [8.2, 21.0, 86., 221.], [0, 1, 2, 3])"
srun --cpu-bind=none python -c "from hist_dirac import DiracHistograms; DiracHistograms('20231216dirac', [21.0, 86.], [0, 1, 2, 3])"