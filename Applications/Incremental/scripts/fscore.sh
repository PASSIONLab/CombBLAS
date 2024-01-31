#!/bin/bash -l

#SBATCH -q regular 
#SBATCH -C knl
##SBATCH -A m1982 # Aydin's project (PASSION: Scalable Graph Learning for Scientific Discovery)
##SBATCH -A m2865 # Exabiome project
#SBATCH -A m4012 # Ariful's project (GraphML: Intelligent Premitives for Graph Machine Learning)

#SBATCH -t 4:00:00

#SBATCH -N 1
#SBATCH -J fscore
#SBATCH -o slurm.fscore.o%j

### For Perlmutter
#SYSTEM=perlmutter_cpu
#N_NODE=4
#PROC_PER_NODE=16
#N_PROC=64
#THREAD_PER_PROC=8
#PER_PROC_MEM=30
#export OMP_NUM_THREADS=$THREAD_PER_PROC

## For Cori 
SYSTEM=cori_knl
N_NODE=1
PROC_PER_NODE=1
N_PROC=1
THREAD_PER_PROC=64
export OMP_NUM_THREADS=$THREAD_PER_PROC

#PREFIX=$SCRATCH/virus-incremental/20-split/virus_30_50
#DATA_NAME=virus
#N_SPLIT=20

#PREFIX=$SCRATCH/eukarya-incremental/50-split/eukarya_30_50_length
#DATA_NAME=eukarya
#N_SPLIT=50


PREFIX=$SCRATCH/geom-incremental/3-split/geom
DATA_NAME=geom
N_SPLIT=3

BINARY=$HOME/Codes/CombBLAS/Applications/Incremental/fscore
START_STEP=0
END_STEP=$(($N_SPLIT - 1))
ALG_TEST=v33
ALG_GT=full
for STEP in $(seq $START_STEP $END_STEP);
    do echo $STEP; 
    CLUSTER_FILE_GT=$PREFIX.$N_SPLIT.$STEP.$ALG_GT
    CLUSTER_FILE_TEST=$PREFIX.$N_SPLIT.$STEP.$ALG_TEST
    FSCORE_FILE=$CLUSTER_FILE_TEST.fscore
    echo GT:$CLUSTER_FILE_GT
    echo TEST:$CLUSTER_FILE_TEST
    $BINARY -M1 $CLUSTER_FILE_GT -M2 $CLUSTER_FILE_TEST -base 0 &> $FSCORE_FILE
    echo ---
done
