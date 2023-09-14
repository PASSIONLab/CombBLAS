#!/bin/bash -l

#SBATCH -q debug 
#SBATCH -C cpu
##SBATCH -A m1982 # Aydin's project (PASSION: Scalable Graph Learning for Scientific Discovery)
##SBATCH -A m2865 # Exabiome project
##SBATCH -A m4012 # Ariful's project (GraphML: Intelligent Premitives for Graph Machine Learning)
#SBATCH -A m4293 # Sparsitute project (A Mathematical Institute for Sparse Computations in Science and Engineering)

#SBATCH -t 0:30:00

#SBATCH -N 4
#SBATCH -J lcc
#SBATCH -o slurm.lcc.o%j

N_NODE=4
PROC_PER_NODE=16
N_PROC=$(( $N_NODE * $PROC_PER_NODE ))
THREAD_PER_PROC=8
export OMP_NUM_THREADS=$THREAD_PER_PROC

BINARY=$HOME/Codes/CombBLAS/_build/Applications/Incremental/lcc

#DATA_NAME=virus
#IN_FILE=$CFS/m1982/HipMCL/viruses/Renamed_vir_vs_vir_30_50length.indexed.mtx
#OUT_FILE=$CFS/m1982/taufique/virus-lcc/lcc_Renamed_vir_vs_vir_30_50length.indexed.mtx

DATA_NAME=eukarya
IN_FILE=$CFS/m1982/HipMCL/eukarya/Renamed_euk_vs_euk_30_50length.indexed.mtx
OUT_FILE=$CFS/m1982/taufique/eukarya-lcc/lcc_Renamed_euk_vs_euk_30_50length.indexed.mtx

#DATA_NAME=isolates-subgraph1
#IN_FILE=$CFS/m1982/HipMCL/iso_m100/subgraph1/subgraph1_iso_vs_iso_30_70length_ALL.m100.indexed.mtx
#OUT_FILE=$CFS/m1982/taufique/isolates-subgraph1-lcc/lcc_subgraph1_iso_vs_iso_30_70length_ALL.m100.indexed.mtx

srun -N $N_NODE -n $N_PROC -c $THREAD_PER_PROC --ntasks-per-node=$PROC_PER_NODE --cpu-bind=cores \
    $BINARY --infile $IN_FILE --infile-type mm --outfile $OUT_FILE
