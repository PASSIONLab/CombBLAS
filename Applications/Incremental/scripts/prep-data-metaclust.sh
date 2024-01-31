#!/bin/bash -l

#SBATCH -q regular 
#SBATCH -C cpu
##SBATCH -A m1982 # Aydin's project (PASSION: Scalable Graph Learning for Scientific Discovery)
##SBATCH -A m2865 # Exabiome project
##SBATCH -A m4012 # Ariful's project (GraphML: Intelligent Premitives for Graph Machine Learning)
#SBATCH -A m4293 # Sparsitute project (A Mathematical Institute for Sparse Computations in Science and Engineering)

#SBATCH -t 01:30:00

#SBATCH -N 128
#SBATCH -J prep-data-metaclust
#SBATCH -o slurm.prep-data-metaclust.o%j

SYSTEM=perlmutter_cpu
CORE_PER_NODE=128 # Never change. Specific to the system
PER_NODE_MEMORY=256 # Never change. Specific to the system
N_NODE=128
PROC_PER_NODE=8
N_PROC=$(( $N_NODE * $PROC_PER_NODE ))
THREAD_PER_PROC=$(( $CORE_PER_NODE / $PROC_PER_NODE ))
PER_PROC_MEM=$(( $PER_NODE_MEMORY / $PROC_PER_NODE - 2)) #2GB margin of error
export OMP_NUM_THREADS=$THREAD_PER_PROC

#DATA_NAME=metaclust
#IN_FILE=$CFS/m1982/www/HipMCL/Metaclust/Renamed_graph_Metaclust50_MATRIX_DENSE.txt
#N_SPLIT=4
#OUT_PREFIX=$CFS/m1982/taufique/$DATA_NAME-incremental/$N_SPLIT-split/metaclust
#START_SPLIT=0
#END_SPLIT=4

#DATA_NAME=metaclust-subgraph
#IN_FILE=$CFS/m1982/taufique/metaclust-subgraph/metaclust-subgraph.mtx
#N_SPLIT=100
#OUT_PREFIX=$CFS/m1982/taufique/$DATA_NAME-incremental/99-1-split/metaclust-subgraph
#START_SPLIT=0
#END_SPLIT=100

#BINARY=$HOME/Codes/CombBLAS/_build/Applications/Incremental/prep-data-metaclust
#srun -N $N_NODE -n $N_PROC -c $THREAD_PER_PROC --ntasks-per-node=$PROC_PER_NODE --cpu-bind=cores \
    #$BINARY -I mm -M $IN_FILE -out-prefix $OUT_PREFIX \
    #-num-split $N_SPLIT -split-start $START_SPLIT -split-end $END_SPLIT

DATA_NAME=metaclust-subgraph

SPLIT_NAME=99-1
INPUT_DIR=$CFS/m1982/taufique/$DATA_NAME-incremental/$SPLIT_NAME-split
INFILE_PREFIX=metaclust-subgraph-$SPLIT_NAME
OUTPUT_DIR_NAME=hipmcl.$DATA_NAME.$SPLIT_NAME.m11.$SYSTEM.node_"$N_NODE".proc_"$PROC_PER_NODE".thread_"$THREAD_PER_PROC"
OUTPUT_DIR=$SCRATCH/$OUTPUT_DIR_NAME/
if [ -d $OUTPUT_DIR ]; then 
    rm -rf $OUTPUT_DIR; 
fi
mkdir -p $OUTPUT_DIR
STDOUT_FILE=$SCRATCH/stdout.$OUTPUT_DIR_NAME
MAT_FILENAME=$INPUT_DIR/$INFILE_PREFIX.m11.mtx
LBL_FILENAME=$INPUT_DIR/$INFILE_PREFIX.m11.lbl
MSO_FILENAME=$OUTPUT_DIR/$INFILE_PREFIX.m11-summary.mtx
CO_FILENAME=$OUTPUT_DIR/$INFILE_PREFIX.m11.cluster

#INPUT_DIR=$CFS/m1982/taufique/$DATA_NAME
#INFILE_PREFIX=metaclust-subgraph
#OUTPUT_DIR_NAME=hipmcl.$DATA_NAME.$SYSTEM.node_"$N_NODE".proc_"$PROC_PER_NODE".thread_"$THREAD_PER_PROC"
#OUTPUT_DIR=$SCRATCH/$OUTPUT_DIR_NAME/
#if [ -d $OUTPUT_DIR ]; then 
    #rm -rf $OUTPUT_DIR; 
#fi
#mkdir -p $OUTPUT_DIR
#STDOUT_FILE=$SCRATCH/stdout.$OUTPUT_DIR_NAME
#MAT_FILENAME=$INPUT_DIR/$INFILE_PREFIX.mtx
#LBL_FILENAME=$INPUT_DIR/$INFILE_PREFIX.lbl
#MSO_FILENAME=$OUTPUT_DIR/$INFILE_PREFIX.summary.mtx
#CO_FILENAME=$OUTPUT_DIR/$INFILE_PREFIX.cluster

BINARY=$HOME/Codes/CombBLAS/_build/Applications/Incremental/full
if [ -d $OUTPUT_DIR ]; then 
    echo Output directory exists: $OUTPUT_DIR; 
else
    echo Output directory does not exist: $OUTPUT_DIR; 
fi
if [ -f $MAT_FILENAME ]; then 
    echo Input matrix file exists: $MAT_FILENAME
else
    echo Input matrix file does not exist: $MAT_FILENAME
fi
if [ -f $LBL_FILENAME ]; then 
    echo Input label file exists: $LBL_FILENAME
else
    echo Input lbl file does not exist: $LBL_FILENAME
fi
srun -N $N_NODE -n $N_PROC -c $THREAD_PER_PROC --ntasks-per-node=$PROC_PER_NODE --cpu-bind=cores \
    $BINARY -M $MAT_FILENAME -label $LBL_FILENAME -summary-out $MSO_FILENAME -cluster-out $CO_FILENAME -per-process-mem $PER_PROC_MEM &> $STDOUT_FILE
