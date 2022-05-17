#!/bin/bash

#SBATCH --job-name=HPMLproject
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --time=05:00:00
#SBATCH --mem=64GB
#SBATCH --partition=mi50
#SBATCH --output=test2.out



module purge
module load python/intel/3.8.6
module load anaconda3/2020.07
module load cuda/11.3.1
module load openmpi/intel/4.1.1

source activate mpienv

cd /scratch/hh2537/pytorch


mpirun -n 2 python ./test.py 

