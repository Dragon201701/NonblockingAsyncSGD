#!/bin/bash

#SBATCH --job-name=HPMLproject
#SBATCH --nodes=5
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --time=05:00:00
#SBATCH --mem=64GB
#SBATCH --partition=mi50
#SBATCH --output=run.txt



module purge
module load python/intel/3.8.6
module load anaconda3/2020.07
module load cuda/11.3.1
module load openmpi/intel/4.1.1

source activate mpienv

cd /scratch/hh2537/pytorch

num_worker=5
n_step=300
out_file=logs/lab4_$num_worker-$n_step.out 

mpirun -n $num_worker python main.py

