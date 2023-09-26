#!/bin/bash -l

#SBATCH -q regular 
#SBATCH -C cpu
##SBATCH -A m1982 # Aydin's project (PASSION: Scalable Graph Learning for Scientific Discovery)
#SBATCH -A m2865 # Exabiome project
##SBATCH -A m4012 # Ariful's project (GraphML: Intelligent Premitives for Graph Machine Learning)
##SBATCH -A m4293 # Sparsitute project (A Mathematical Institute for Sparse Computations in Science and Engineering)

#SBATCH -t 2:30:00

#SBATCH -N 2
#SBATCH -J inc-pipeline
#SBATCH -o slurm.inc-pipeline.o%j

SYSTEM=perlmutter_cpu
CORE_PER_NODE=128 # Never change. Specific to the system
PER_NODE_MEMORY=256 # Never change. Specific to the system
N_NODE=2
PROC_PER_NODE=8
N_PROC=$(( $N_NODE * $PROC_PER_NODE ))
THREAD_PER_PROC=$(( $CORE_PER_NODE / $PROC_PER_NODE ))
PER_PROC_MEM=$(( $PER_NODE_MEMORY / $PROC_PER_NODE - 2)) #2GB margin of error
export OMP_NUM_THREADS=$THREAD_PER_PROC

#IN_FILE=$SCRATCH/geom/geom.mtx
#OUT_PREFIX=$SCRATCH/geom-incremental/3-split/geom
#DATA_NAME=geom
#N_SPLIT=3

BINARY=$HOME/Codes/CombBLAS/_build/Applications/Incremental/inc-pipeline

DATA_NAME=virus-lcc
N_SPLIT=10
INCREMENTAL_START=10
SUMMARY_THRESHOLD=70
SELECTIVE_PRUNE_THRESHOLD=10 
INPUT_DIR=$CFS/m1982/taufique/virus-lcc-incremental/"$N_SPLIT"-split/
INFILE_PREFIX=lcc_virus_30_50_length

#DATA_NAME=virus
#N_SPLIT=10
#INCREMENTAL_START=10
#SUMMARY_THRESHOLD=70
#SELECTIVE_PRUNE_THRESHOLD=10 
#INPUT_DIR=$CFS/m1982/taufique/virus-incremental/"$N_SPLIT"-split/
#INFILE_PREFIX=vir_30_50_length

#DATA_NAME=eukarya-lcc
#N_SPLIT=50
#INCREMENTAL_START=50 
#SUMMARY_THRESHOLD=70 
#SELECTIVE_PRUNE_THRESHOLD=10 
#INPUT_DIR=$CFS/m1982/taufique/eukarya-lcc-incremental/"$N_SPLIT"-split/
#INFILE_PREFIX=lcc_eukarya_30_50_length

#DATA_NAME=eukarya
#N_SPLIT=50
#INCREMENTAL_START=50 
#SUMMARY_THRESHOLD=70 
#SELECTIVE_PRUNE_THRESHOLD=10 
#INPUT_DIR=$CFS/m1982/taufique/eukarya-incremental/"$N_SPLIT"-split/
#INFILE_PREFIX=eukarya_30_50_length

OUTPUT_DIR_NAME=incremental.$DATA_NAME.$N_SPLIT.$INCREMENTAL_START.$SUMMARY_THRESHOLD.$SELECTIVE_PRUNE_THRESHOLD.$SYSTEM.node_"$N_NODE".proc_"$PROC_PER_NODE".thread_"$THREAD_PER_PROC"
OUTPUT_DIR=$SCRATCH/$OUTPUT_DIR_NAME/
if [ -d $OUTPUT_DIR ]; then 
    rm -rf $OUTPUT_DIR; 
fi
mkdir -p $OUTPUT_DIR
STDOUT_FILE=$SCRATCH/stdout.$OUTPUT_DIR_NAME

echo Running incremental pipeline simulation: $BINARY
echo Using $N_NODE nodes: $N_PROC processes x $THREAD_PER_PROC threads
echo Input directory: $INPUT_DIR
echo Input fileprefix: $INFILE_PREFIX
echo Number of splits: $N_SPLIT
echo Incremental start step: $INCREMENTAL_START
echo Summary threshold: $SUMMARY_THRESHOLD
echo Selective prune threshold: $SELECTIVE_PRUNE_THRESHOLD
echo Output directory: $OUTPUT_DIR
echo Per process memory: $PER_PROC_MEM GB

srun -N $N_NODE -n $N_PROC -c $THREAD_PER_PROC --ntasks-per-node=$PROC_PER_NODE --cpu-bind=cores \
    $BINARY \
    --input-dir $INPUT_DIR --infile-prefix $INFILE_PREFIX \
    --num-split $N_SPLIT --incremental-start $INCREMENTAL_START --summary-threshold $SUMMARY_THRESHOLD --selective-prune-threshold $SELECTIVE_PRUNE_THRESHOLD\
    --hipmcl-before-incremental 1\
    --output-dir $OUTPUT_DIR --per-process-mem $PER_PROC_MEM &> $STDOUT_FILE
echo ---

for INCREMENTAL_START in 4 6 8 # For virus
#for INCREMENTAL_START in 40 # For eukarya 20, 30, 40
do
    for SUMMARY_THRESHOLD in 30 50 70 90 # 30%, 50%, 70%, 90%
    do
        for SELECTIVE_PRUNE_THRESHOLD in 5 10 30 50 70 # 5% 10%, 30%, 50%, 70%
        do
            OUTPUT_DIR_NAME=incremental.$DATA_NAME.$N_SPLIT.$INCREMENTAL_START.$SUMMARY_THRESHOLD.$SELECTIVE_PRUNE_THRESHOLD.$SYSTEM.node_"$N_NODE".proc_"$PROC_PER_NODE".thread_"$THREAD_PER_PROC"
            OUTPUT_DIR=$SCRATCH/$OUTPUT_DIR_NAME/
            if [ -d $OUTPUT_DIR ]; then 
                rm -rf $OUTPUT_DIR; 
            fi
            mkdir -p $OUTPUT_DIR
            STDOUT_FILE=$SCRATCH/stdout.$OUTPUT_DIR_NAME

            echo Running incremental pipeline simulation: $BINARY
            echo Using $N_NODE nodes: $N_PROC processes x $THREAD_PER_PROC threads
            echo Input directory: $INPUT_DIR
            echo Input fileprefix: $INFILE_PREFIX
            echo Number of splits: $N_SPLIT
            echo Incremental start step: $INCREMENTAL_START
            echo Summary threshold: $SUMMARY_THRESHOLD
            echo Selective prune threshold: $SELECTIVE_PRUNE_THRESHOLD
            echo Output directory: $OUTPUT_DIR
            echo Per process memory: $PER_PROC_MEM GB

            srun -N $N_NODE -n $N_PROC -c $THREAD_PER_PROC --ntasks-per-node=$PROC_PER_NODE --cpu-bind=cores \
                $BINARY \
                --input-dir $INPUT_DIR --infile-prefix $INFILE_PREFIX \
                --num-split $N_SPLIT --incremental-start $INCREMENTAL_START --summary-threshold $SUMMARY_THRESHOLD --selective-prune-threshold $SELECTIVE_PRUNE_THRESHOLD\
                --output-dir $OUTPUT_DIR --per-process-mem $PER_PROC_MEM &> $STDOUT_FILE
            echo ---

        done
    done
done

