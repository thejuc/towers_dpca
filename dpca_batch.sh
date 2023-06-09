#!/usr/bin/env bash


#SBATCH -o slurm-%j.out
#SBATCH -p all
#SBATCH --output=logs/%u_%x_%A.out
#SBATCH --mail-type=END
#SBATCH --cpus-per-task=1
#SBATCH --nodes=1
#SBATCH --ntasks=1

echo "In the directory: `pwd` "
echo "As the user: `whoami` "
echo "on host: `hostname` "

echo "With access to cpu id(s): "
cat /proc/$$/status | grep Cpus_allowed_list

echo "Array Allocation Number: $SLURM_ARRAY_JOB_ID"
echo "Array Index: $SLURM_ARRAY_TASK_ID"

module load anacondapy
source activate witten_rotation

python /jukebox/witten/yousuf/rotation/towers_dpca_repo/dpca_run_all.py $1
conda deactivate 