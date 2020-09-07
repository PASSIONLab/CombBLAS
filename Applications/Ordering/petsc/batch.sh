#!/bin/bash -l

#SBATCH -p debug
#SBATCH -N 43
#SBATCH -t 00:30:00
#SBATCH -J petsc_cg
#SBATCH -o petsc_cg.o%j

export OMP_NUM_THREADS=1
MAT=parabolic_fem.bin
srun -n 1 -N 1 ./ex18 -f0 $MAT -ksp_type cg
srun -n 2 -N 1 ./ex18 -f0 $MAT -ksp_type cg
srun -n 4 -N 1 ./ex18 -f0 $MAT -ksp_type cg
srun -n 8 -N 1 ./ex18 -f0 $MAT -ksp_type cg
srun -n 16 -N 1 ./ex18 -f0 $MAT -ksp_type cg
srun -n 32 -N 2 ./ex18 -f0 $MAT -ksp_type cg
srun -n 64 -N 3 ./ex18 -f0 $MAT -ksp_type cg
srun -n 128 -N 6 ./ex18 -f0 $MAT -ksp_type cg
srun -n 256 -N 11 ./ex18 -f0 $MAT -ksp_type cg
srun -n 512 -N 22 ./ex18 -f0 $MAT -ksp_type cg
srun -n 1024 -N 43 ./ex18 -f0 $MAT -ksp_type cg


MAT=parabolic_fem_rcm.bin
srun -n 1 -N 1 ./ex18 -f0 $MAT -ksp_type cg
srun -n 2 -N 1 ./ex18 -f0 $MAT -ksp_type cg
srun -n 4 -N 1 ./ex18 -f0 $MAT -ksp_type cg
srun -n 8 -N 1 ./ex18 -f0 $MAT -ksp_type cg
srun -n 16 -N 1 ./ex18 -f0 $MAT -ksp_type cg
srun -n 32 -N 2 ./ex18 -f0 $MAT -ksp_type cg
srun -n 64 -N 3 ./ex18 -f0 $MAT -ksp_type cg
srun -n 128 -N 6 ./ex18 -f0 $MAT -ksp_type cg
srun -n 256 -N 11 ./ex18 -f0 $MAT -ksp_type cg
srun -n 512 -N 22 ./ex18 -f0 $MAT -ksp_type cg
srun -n 1024 -N 43 ./ex18 -f0 $MAT -ksp_type cg


MAT=thermal2.bin
srun -n 1 -N 1 ./ex18 -f0 $MAT -ksp_type cg
srun -n 2 -N 1 ./ex18 -f0 $MAT -ksp_type cg
srun -n 4 -N 1 ./ex18 -f0 $MAT -ksp_type cg
srun -n 8 -N 1 ./ex18 -f0 $MAT -ksp_type cg
srun -n 16 -N 1 ./ex18 -f0 $MAT -ksp_type cg
srun -n 32 -N 2 ./ex18 -f0 $MAT -ksp_type cg
srun -n 64 -N 3 ./ex18 -f0 $MAT -ksp_type cg
srun -n 128 -N 6 ./ex18 -f0 $MAT -ksp_type cg
srun -n 256 -N 11 ./ex18 -f0 $MAT -ksp_type cg
srun -n 512 -N 22 ./ex18 -f0 $MAT -ksp_type cg
srun -n 1024 -N 43 ./ex18 -f0 $MAT -ksp_type cg


MAT=thermal2_rcm.bin
srun -n 1 -N 1 ./ex18 -f0 $MAT -ksp_type cg
srun -n 2 -N 1 ./ex18 -f0 $MAT -ksp_type cg
srun -n 4 -N 1 ./ex18 -f0 $MAT -ksp_type cg
srun -n 8 -N 1 ./ex18 -f0 $MAT -ksp_type cg
srun -n 16 -N 1 ./ex18 -f0 $MAT -ksp_type cg
srun -n 32 -N 2 ./ex18 -f0 $MAT -ksp_type cg
srun -n 64 -N 3 ./ex18 -f0 $MAT -ksp_type cg
srun -n 128 -N 6 ./ex18 -f0 $MAT -ksp_type cg
srun -n 256 -N 11 ./ex18 -f0 $MAT -ksp_type cg
srun -n 512 -N 22 ./ex18 -f0 $MAT -ksp_type cg
srun -n 1024 -N 43 ./ex18 -f0 $MAT -ksp_type cg



MAT=af_shell4.bin
srun -n 1 -N 1 ./ex18 -f0 $MAT -ksp_type cg
srun -n 2 -N 1 ./ex18 -f0 $MAT -ksp_type cg
srun -n 4 -N 1 ./ex18 -f0 $MAT -ksp_type cg
srun -n 8 -N 1 ./ex18 -f0 $MAT -ksp_type cg
srun -n 16 -N 1 ./ex18 -f0 $MAT -ksp_type cg
srun -n 32 -N 2 ./ex18 -f0 $MAT -ksp_type cg
srun -n 64 -N 3 ./ex18 -f0 $MAT -ksp_type cg
srun -n 128 -N 6 ./ex18 -f0 $MAT -ksp_type cg
srun -n 256 -N 11 ./ex18 -f0 $MAT -ksp_type cg
srun -n 512 -N 22 ./ex18 -f0 $MAT -ksp_type cg
srun -n 1024 -N 43 ./ex18 -f0 $MAT -ksp_type cg



MAT=af_shell4_rcm.bin
srun -n 1 -N 1 ./ex18 -f0 $MAT -ksp_type cg
srun -n 2 -N 1 ./ex18 -f0 $MAT -ksp_type cg
srun -n 4 -N 1 ./ex18 -f0 $MAT -ksp_type cg
srun -n 8 -N 1 ./ex18 -f0 $MAT -ksp_type cg
srun -n 16 -N 1 ./ex18 -f0 $MAT -ksp_type cg
srun -n 32 -N 2 ./ex18 -f0 $MAT -ksp_type cg
srun -n 64 -N 3 ./ex18 -f0 $MAT -ksp_type cg
srun -n 128 -N 6 ./ex18 -f0 $MAT -ksp_type cg
srun -n 256 -N 11 ./ex18 -f0 $MAT -ksp_type cg
srun -n 512 -N 22 ./ex18 -f0 $MAT -ksp_type cg
srun -n 1024 -N 43 ./ex18 -f0 $MAT -ksp_type cg

