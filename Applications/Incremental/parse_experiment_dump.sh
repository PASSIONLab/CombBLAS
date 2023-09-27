#!/bin/bash -l

SYSTEM=perlmutter_cpu
CORE_PER_NODE=128 # Never change. Specific to the system
PER_NODE_MEMORY=256 # Never change. Specific to the system

#IN_FILE=$SCRATCH/geom/geom.mtx
#OUT_PREFIX=$SCRATCH/geom-incremental/3-split/geom
#DATA_NAME=geom
#N_SPLIT=3

PYSCRIPT=$HOME/Codes/CombBLAS/Applications/Incremental/parse_experiment_dump.py

#DATA_NAME=virus-lcc
#N_SPLIT=10
#INCREMENTAL_START=10
#SUMMARY_THRESHOLD=70
#SELECTIVE_PRUNE_THRESHOLD=10 
#INPUT_DIR=$CFS/m1982/taufique/virus-lcc-incremental/"$N_SPLIT"-split/
#INFILE_PREFIX=lcc_virus_30_50_length
#N_NODE=2
#PROC_PER_NODE=8

#DATA_NAME=virus
#N_SPLIT=10
#INCREMENTAL_START=10
#SUMMARY_THRESHOLD=70
#SELECTIVE_PRUNE_THRESHOLD=10 
#INPUT_DIR=$CFS/m1982/taufique/virus-incremental/"$N_SPLIT"-split/
#INFILE_PREFIX=vir_30_50_length
#N_NODE=2
#PROC_PER_NODE=8

#DATA_NAME=eukarya-lcc
#N_SPLIT=50
#INCREMENTAL_START=50 
#SUMMARY_THRESHOLD=70 
#SELECTIVE_PRUNE_THRESHOLD=10 
#INPUT_DIR=$CFS/m1982/taufique/eukarya-lcc-incremental/"$N_SPLIT"-split/
#INFILE_PREFIX=lcc_eukarya_30_50_length
#N_NODE=8
#PROC_PER_NODE=8

DATA_NAME=eukarya
N_SPLIT=50
INCREMENTAL_START=50 
SUMMARY_THRESHOLD=70 
SELECTIVE_PRUNE_THRESHOLD=10 
INPUT_DIR=$CFS/m1982/taufique/eukarya-incremental/"$N_SPLIT"-split/
INFILE_PREFIX=eukarya_30_50_length
N_NODE=8
PROC_PER_NODE=8

N_PROC=$(( $N_NODE * $PROC_PER_NODE ))
THREAD_PER_PROC=$(( $CORE_PER_NODE / $PROC_PER_NODE ))
PER_PROC_MEM=$(( $PER_NODE_MEMORY / $PROC_PER_NODE - 2)) #2GB margin of error
export OMP_NUM_THREADS=$THREAD_PER_PROC

INPUT_FILE_SUFFIX=incremental.$DATA_NAME.$N_SPLIT.$INCREMENTAL_START.$SUMMARY_THRESHOLD.$SELECTIVE_PRUNE_THRESHOLD.$SYSTEM.node_"$N_NODE".proc_"$PROC_PER_NODE".thread_"$THREAD_PER_PROC"
INPUT_FILE=$SCRATCH/stdout.$INPUT_FILE_SUFFIX
INPUT_FILE=$SCRATCH/stdout.$INPUT_FILE_SUFFIX
OUTPUT_FILE=$SCRATCH/csv.$INPUT_FILE_SUFFIX

echo Running: $PYSCRIPT
echo Data: $DATA_NAME
echo Input file: $INPUT_FILE
echo Output file: $OUTPUT_FILE

python $PYSCRIPT $DATA_NAME $INPUT_FILE $OUTPUT_FILE

echo ---

#for INCREMENTAL_START in 4 6 8 # For virus
for INCREMENTAL_START in 20 30 40 # For eukarya 20, 30, 40
do
    for SUMMARY_THRESHOLD in 30 50 70 90 # 30%, 50%, 70%, 90%
    do
        for SELECTIVE_PRUNE_THRESHOLD in 5 10 30 50 70 # 5% 10%, 30%, 50%, 70%
        do
            N_PROC=$(( $N_NODE * $PROC_PER_NODE ))
            THREAD_PER_PROC=$(( $CORE_PER_NODE / $PROC_PER_NODE ))
            PER_PROC_MEM=$(( $PER_NODE_MEMORY / $PROC_PER_NODE - 2)) #2GB margin of error
            export OMP_NUM_THREADS=$THREAD_PER_PROC

            INPUT_FILE_SUFFIX=incremental.$DATA_NAME.$N_SPLIT.$INCREMENTAL_START.$SUMMARY_THRESHOLD.$SELECTIVE_PRUNE_THRESHOLD.$SYSTEM.node_"$N_NODE".proc_"$PROC_PER_NODE".thread_"$THREAD_PER_PROC"
            INPUT_FILE=$SCRATCH/stdout.$INPUT_FILE_SUFFIX
            INPUT_FILE=$SCRATCH/stdout.$INPUT_FILE_SUFFIX
            OUTPUT_FILE=$SCRATCH/csv.$INPUT_FILE_SUFFIX

            echo Running: $PYSCRIPT
            echo Data: $DATA_NAME
            echo Input file: $INPUT_FILE
            echo Output file: $OUTPUT_FILE

            python $PYSCRIPT $DATA_NAME $INPUT_FILE $OUTPUT_FILE

            echo ---

        done
    done
done

