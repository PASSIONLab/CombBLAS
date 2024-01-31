#!/bin/bash -l

#SBATCH -q regular 
#SBATCH -C cpu
##SBATCH -A m1982 # Aydin's project (PASSION: Scalable Graph Learning for Scientific Discovery)
##SBATCH -A m2865 # Exabiome project
##SBATCH -A m4012 # Ariful's project (GraphML: Intelligent Premitives for Graph Machine Learning)
#SBATCH -A m4293 # Sparsitute project (A Mathematical Institute for Sparse Computations in Science and Engineering)

#SBATCH -t 0:30:00

#SBATCH -N 32
#SBATCH -J inc
#SBATCH -o slurm.inc.o%j

SYSTEM=perlmutter_cpu
CORE_PER_NODE=128 # Never change. Specific to the system
PER_NODE_MEMORY=256 # Never change. Specific to the system
N_NODE=32
PROC_PER_NODE=8
N_PROC=$(( $N_NODE * $PROC_PER_NODE ))
THREAD_PER_PROC=$(( $CORE_PER_NODE / $PROC_PER_NODE ))
PER_PROC_MEM=$(( $PER_NODE_MEMORY / $PROC_PER_NODE - 2)) #2GB margin of error
export OMP_NUM_THREADS=$THREAD_PER_PROC

BINARY=$HOME/Codes/CombBLAS/_build/Applications/Incremental/inc

DATA_NAME=metaclust-subgraph
SPLIT_NAME=99-1
INPUT_DIR=$CFS/m1982/taufique/$DATA_NAME-incremental/$SPLIT_NAME-split
INFILE_PREFIX=metaclust-subgraph-$SPLIT_NAME
SUMMARY_THRESHOLD=50 
SELECTIVE_PRUNE_THRESHOLD=10 
ALG=inc

OUTPUT_DIR_NAME=$ALG.$DATA_NAME.$SPLIT_NAME.$SUMMARY_THRESHOLD.$SELECTIVE_PRUNE_THRESHOLD.$SYSTEM.node_"$N_NODE".proc_"$PROC_PER_NODE".thread_"$THREAD_PER_PROC"
OUTPUT_DIR=$SCRATCH/$OUTPUT_DIR_NAME/
if [ -d $OUTPUT_DIR ]; then 
    rm -rf $OUTPUT_DIR; 
fi
mkdir -p $OUTPUT_DIR
STDOUT_FILE=$SCRATCH/stdout.$OUTPUT_DIR_NAME

M11_FILE=$INPUT_DIR/$INFILE_PREFIX.m11-summary.mtx
M12_FILE=$INPUT_DIR/$INFILE_PREFIX.m12.mtx
M21_FILE=$INPUT_DIR/$INFILE_PREFIX.m21.mtx
M22_FILE=$INPUT_DIR/$INFILE_PREFIX.m22.mtx
L11_FILE=$INPUT_DIR/$INFILE_PREFIX.m11.lbl
L22_FILE=$INPUT_DIR/$INFILE_PREFIX.m22.lbl
SUMMARY_THRESHOLD=50 
SELECTIVE_PRUNE_THRESHOLD=10 
MINC_FILE=$OUTPUT_DIR/$INFILE_PREFIX.minc.mtx
MSO_FILE=$OUTPUT_DIR/$INFILE_PREFIX.summary.mtx
LO_FILE=$OUTPUT_DIR/$INFILE_PREFIX.minc.lbl
CO_FILE=$OUTPUT_DIR/$INFILE_PREFIX.minc.cluster

if [ -d $OUTPUT_DIR ]; then 
    echo [Y] Output directory: $OUTPUT_DIR; 
else
    echo [N] Output directory: $OUTPUT_DIR; 
fi
if [ -f $M11_FILE ]; then 
    echo [Y] M11 matrix file: $M11_FILE
else
    echo [N] M11 matrix file: $M11_FILE
fi
if [ -f $M12_FILE ]; then 
    echo [Y] M12 matrix file: $M12_FILE
else
    echo [N] M12 matrix file: $M12_FILE
fi
if [ -f $M21_FILE ]; then 
    echo [Y] M21 matrix file: $M21_FILE
else
    echo [N] M21 matrix file: $M21_FILE
fi
if [ -f $M22_FILE ]; then 
    echo [Y] M22 matrix file: $M22_FILE
else
    echo [N] M22 matrix file: $M22_FILE
fi
if [ -f $L11_FILE ]; then 
    echo [Y] L11 label file: $L11_FILE
else
    echo [N] L11 label file: $L11_FILE
fi
if [ -f $L22_FILE ]; then 
    echo [Y] L22 label file: $L22_FILE
else
    echo [N] L22 label file: $L22_FILE
fi

srun -N $N_NODE -n $N_PROC -c $THREAD_PER_PROC --ntasks-per-node=$PROC_PER_NODE --cpu-bind=cores $BINARY \
    --M11 $M11_FILE \
    --M12 $M12_FILE \
    --M21 $M21_FILE \
    --M22 $M22_FILE \
    --L11 $L11_FILE \
    --L22 $L22_FILE \
    --summary-threshold $SUMMARY_THRESHOLD \
    --selective-prune-threshold $SELECTIVE_PRUNE_THRESHOLD \
    --inc-mat-out $MINC_FILE \
    --summary-out $MSO_FILE \
    --label-out $LO_FILE \
    --cluster-out $CO_FILE \
    --alg $ALG \
    --per-process-mem $PER_PROC_MEM &> $STDOUT_FILE
