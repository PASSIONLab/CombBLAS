#!/bin/bash -l

#SBATCH -q debug 
#SBATCH -C cpu
##SBATCH -A m1982 # Aydin's project (PASSION: Scalable Graph Learning for Scientific Discovery)
##SBATCH -A m2865 # Exabiome project
##SBATCH -A m4012 # Ariful's project (GraphML: Intelligent Premitives for Graph Machine Learning)
#SBATCH -A m4293 # Sparsitute project (A Mathematical Institute for Sparse Computations in Science and Engineering)

#SBATCH -t 0:30:00

#SBATCH -N 2
#SBATCH -J petsc_test
#SBATCH -o slurm.petsc_test.o%j

SYSTEM=perlmutter_cpu
CORE_PER_NODE=128 # Never change. Specific to the system
PER_NODE_MEMORY=256 # Never change. Specific to the system
N_NODE=2
PROC_PER_NODE=8
N_PROC=$(( $N_NODE * $PROC_PER_NODE ))
THREAD_PER_PROC=$(( $CORE_PER_NODE / $PROC_PER_NODE ))
PER_PROC_MEM=$(( $PER_NODE_MEMORY / $PROC_PER_NODE - 2)) #2GB margin of error
export OMP_NUM_THREADS=$THREAD_PER_PROC

A_FILE=$CFS/m4293/datasets/uk-2002/uk-2002.petsc
B_FILE=$CFS/m4293/datasets/uk-2002/sp_80/d_128/sparse_local.petsc
C_FILE=$CFS/m4293/datasets/uk-2002/sp_80/d_128/c.petsc
#A_FILE=$SCRATCH/spgemm_summa/uk-2002.mtx
#B_FILE=$SCRATCH/spgemm_summa/uk-2002.mtx
##B_FILE=$SCRATCH/spgemm_summa/sparse_local.txt
#C_FILE=$SCRATCH/summa_spgemm/c.mtx

#A_FILE=$SCRATCH/TESTDATA/rmat_scale16_A.mtx
#B_FILE=$SCRATCH/TESTDATA/rmat_scale16_B.mtx

BINARY=$HOME/Codes/CombBLAS/_build/Applications/Tall-Skinny-SpGEMM/petsc_test

srun -N $N_NODE -n $N_PROC -c $THREAD_PER_PROC --ntasks-per-node=$PROC_PER_NODE --cpu-bind=cores $BINARY $A_FILE $B_FILE $C_FILE 1
