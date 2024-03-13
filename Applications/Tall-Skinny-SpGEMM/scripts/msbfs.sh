#!/bin/bash -l

#SBATCH -q debug 
#SBATCH -C cpu
##SBATCH -A m1982 # Aydin's project (PASSION: Scalable Graph Learning for Scientific Discovery)
##SBATCH -A m2865 # Exabiome project
##SBATCH -A m4012 # Ariful's project (GraphML: Intelligent Premitives for Graph Machine Learning)
#SBATCH -A m4293 # Sparsitute project (A Mathematical Institute for Sparse Computations in Science and Engineering)

#SBATCH -t 0:30:00

#SBATCH -N 1
#SBATCH -J msbfs
#SBATCH -o slurm.msbfs.o%j

SYSTEM=perlmutter_cpu
CORE_PER_NODE=128 # Never change. Specific to the system
PER_NODE_MEMORY=256 # Never change. Specific to the system
N_NODE=1
PROC_PER_NODE=4
N_PROC=$(( $N_NODE * $PROC_PER_NODE ))
THREAD_PER_PROC=$(( $CORE_PER_NODE / $PROC_PER_NODE ))
PER_PROC_MEM=$(( $PER_NODE_MEMORY / $PROC_PER_NODE - 2)) #2GB margin of error
export OMP_NUM_THREADS=$THREAD_PER_PROC

A_FILE=$HOME/Codes/CombBLAS/Applications/Tall-Skinny-SpGEMM/scripts/toy-data/sevenvertex.mtx
B_FILE=$HOME/Codes/CombBLAS/Applications/Tall-Skinny-SpGEMM/scripts/toy-data/all-source.mtx
BINARY=$HOME/Codes/CombBLAS/_build/Applications/Tall-Skinny-SpGEMM/msbfs

srun -N $N_NODE -n $N_PROC -c $THREAD_PER_PROC --ntasks-per-node=$PROC_PER_NODE --cpu-bind=cores $BINARY -A $A_FILE -B $B_FILE -permute no
