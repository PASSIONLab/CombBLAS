#!/bin/bash -l

#SBATCH -p debug
#SBATCH -N 4
#SBATCH -C knl
#SBATCH -t 00:20:00
#SBATCH -J trilinosMatSquare_knl_4
#SBATCH -o trilinosMatSquare_knl_4.o%j

export OMP_NUM_THREADS=1

MAT=/project/projectdirs/m1982/ariful/SuperLU_matrices/cage12.mtx
N=4
n=256
srun -N $N -n $n -c 1  --ntasks-per-node=64 ./trilinosMatSquare $MAT 


