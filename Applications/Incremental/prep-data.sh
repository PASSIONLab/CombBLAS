#!/bin/bash -l

#SBATCH -q debug 
#SBATCH -C cpu
##SBATCH -A m1982 # Aydin's project (PASSION: Scalable Graph Learning for Scientific Discovery)
##SBATCH -A m2865 # Exabiome project
##SBATCH -A m4012 # Ariful's project (GraphML: Intelligent Premitives for Graph Machine Learning)
#SBATCH -A m4293 # Sparsitute project (A Mathematical Institute for Sparse Computations in Science and Engineering)

#SBATCH -t 0:30:00

#SBATCH -N 1
#SBATCH -J prep-data
#SBATCH -o slurm.prep-data.o%j

SYSTEM=perlmutter_cpu
N_NODE=1
PROC_PER_NODE=16
N_PROC=$(( $N_NODE * $PROC_PER_NODE ))
THREAD_PER_PROC=8
export OMP_NUM_THREADS=$THREAD_PER_PROC

#DATA_NAME=virus
#IN_FILE=$CFS/m1982/HipMCL/viruses/Renamed_vir_vs_vir_30_50length.indexed.mtx
#OUT_PREFIX=$CFS/m1982/taufique/virus-incremental/10-split/vir_30_50_length
#N_SPLIT=10
#START_SPLIT=0
#END_SPLIT=10

#DATA_NAME=virus-lcc
#IN_FILE=$CFS/m1982/taufique/virus-lcc/lcc_Renamed_vir_vs_vir_30_50length.indexed.mtx
#OUT_PREFIX=$CFS/m1982/taufique/virus-lcc-incremental/10-split/lcc_virus_30_50_length
#N_SPLIT=10
#START_SPLIT=0
#END_SPLIT=10

#DATA_NAME=eukarya
#IN_FILE=$CFS/m1982/HipMCL/eukarya/Renamed_euk_vs_euk_30_50length.indexed.mtx
#OUT_PREFIX=$CFS/m1982/taufique/eukarya-incremental/50-split/eukarya_30_50_length
#N_SPLIT=50
#START_SPLIT=0
#END_SPLIT=50

#DATA_NAME=eukarya-lcc
#IN_FILE=$CFS/m1982/taufique/eukarya-lcc/lcc_Renamed_euk_vs_euk_30_50length.indexed.mtx
#OUT_PREFIX=$CFS/m1982/taufique/eukarya-lcc-incremental/50-split/lcc_eukarya_30_50_length
#N_SPLIT=50
#START_SPLIT=0
#END_SPLIT=50

##DATA_NAME=archea
#IN_FILE=$CFS/m1982/HipMCL/archaea/Renamed_arch_vs_arch_30_50length.indexed.mtx
#OUT_PREFIX=$SCRATCH/archea-incremental/50-split/archea_30_50_length
#N_SPLIT=50
#START_SPLIT=0
#END_SPLIT=50

BINARY=$HOME/Codes/CombBLAS/_build/Applications/Incremental/prep-data
srun -N $N_NODE -n $N_PROC -c $THREAD_PER_PROC --ntasks-per-node=$PROC_PER_NODE --cpu-bind=cores \
    $BINARY -I mm -M $IN_FILE -out-prefix $OUT_PREFIX \
    -num-split $N_SPLIT -split-start $START_SPLIT -split-end $END_SPLIT
