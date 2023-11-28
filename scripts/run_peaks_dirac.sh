#!/bin/bash 
#SBATCH -A des 
#SBATCH -C cpu 
#SBATCH -q regular 
#SBATCH -t 10:00:00 
#SBATCH --nodes=4
#SBATCH --cpus-per-task=128
#SBATCH --job-name=run_peaks_dirac
#SBATCH --mem=0
#SBATCH --mail-type=ALL
#SBATCH --mail-user=shubh@sas.upenn.edu

module load python
conda activate gnn
cd /global/cfs/cdirs/des/shubh/graphs/graphs_weak_lensing/scripts
# srun --cpu-bind=none python -c "from peaks_dirac import DiracPatches; DiracPatches('20231115dirac_tomobin0_scale21.0')"
srun --cpu-bind=none python -c "from peaks_dirac import DiracPatches; DiracPatches('20231128dirac', [8.2, 21.0, 86., 221.], [0, 1, 2, 3])"