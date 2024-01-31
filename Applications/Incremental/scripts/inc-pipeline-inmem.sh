#!/bin/bash -l

#SBATCH -q debug 
#SBATCH -C knl
##SBATCH -A m1982 # Aydin's project (PASSION: Scalable Graph Learning for Scientific Discovery)
##SBATCH -A m2865 # Exabiome project
#SBATCH -A m4012 # Ariful's project (GraphML: Intelligent Premitives for Graph Machine Learning)

#SBATCH -t 0:30:00

#SBATCH -N 1
#SBATCH -J inc-pipeline
#SBATCH -o slurm.inc-pipeline.o%j

### For Perlmutter
#SYSTEM=perlmutter_cpu
#export OMP_NUM_THREADS=8
#N_NODE=4
#PROC_PER_NODE=16
#N_PROC=64
#THREAD_PER_PROC=8
#PER_PROC_MEM=30

## For Cori 
SYSTEM=cori_knl
export OMP_NUM_THREADS=1
N_NODE=1
PROC_PER_NODE=1
N_PROC=1
THREAD_PER_PROC=1
PER_PROC_MEM=24

#IN_FILE=$SCRATCH/virus/vir_vs_vir_30_50length_propermm.mtx
#OUT_PREFIX=$SCRATCH/virus-incremental/20-split/virus_30_50
#DATA_NAME=virus
#N_SPLIT=20

#IN_FILE=$SCRATCH/eukarya-debug/Renamed_euk_vs_euk_30_50length.indexed.mtx
#OUT_PREFIX=$SCRATCH/eukarya-incremental/50-split/eukarya_30_50_length
#DATA_NAME=eukarya
#N_SPLIT=50

IN_FILE=$SCRATCH/geom/geom.mtx
OUT_PREFIX=$SCRATCH/geom-incremental/3-split/geom
DATA_NAME=geom
N_SPLIT=3

BINARY=$HOME/Codes/CombBLAS/_build/Applications/Incremental/inc-pipeline
ALG=v33
STDOUT_DIR=$HOME/Codes/CombBLAS/Applications/Incremental/stdout.inc_$ALG.$DATA_NAME.$SYSTEM.node_$N_NODE; mkdir -p $STDOUT_DIR;
STDOUT_FILE=$STDOUT_DIR/stdout.$ALG.pipeline-simulation

echo $STDOUT_FILE

srun -N $N_NODE -n $N_PROC -c $THREAD_PER_PROC --ntasks-per-node=$PROC_PER_NODE --cpu-bind=cores \
    $BINARY -I mm -M $IN_FILE -out-prefix $OUT_PREFIX -num-split $N_SPLIT -alg $ALG -per-process-mem $PER_PROC_MEM &> $STDOUT_FILE
