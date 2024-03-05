#!/bin/bash -l

#SBATCH -q regular
#SBATCH -C cpu
##SBATCH -A m1982 # Aydin's project (PASSION: Scalable Graph Learning for Scientific Discovery)
##SBATCH -A m2865 # Exabiome project
##SBATCH -A m4012 # Ariful's project (GraphML: Intelligent Premitives for Graph Machine Learning)
##SBATCH -A m4293 # Sparsitute project (A Mathematical Institute for Sparse Computations in Science and Engineering)

#SBATCH -t 12:30:00

#SBATCH -N 1
#SBATCH -J abl-fscore
#SBATCH -o slurm.abl-fscore.o%j

SYSTEM=perlmutter_cpu
CORE_PER_NODE=128 # Never change. Specific to the system
PER_NODE_MEMORY=256 # Never change. Specific to the system
N_NODE=8
PROC_PER_NODE=8
N_PROC=$(( $N_NODE * $PROC_PER_NODE ))
THREAD_PER_PROC=$(( $CORE_PER_NODE / $PROC_PER_NODE ))
PER_PROC_MEM=$(( $PER_NODE_MEMORY / $PROC_PER_NODE - 2)) #2GB margin of error
export OMP_NUM_THREADS=$THREAD_PER_PROC

BINARY=$HOME/Codes/CombBLAS/Applications/Incremental/fscore

#DATA_NAME=virus-lcc
#N_SPLIT=10
#INCREMENTAL_START=10
#SUMMARY_THRESHOLD=70
#SELECTIVE_PRUNE_THRESHOLD=10 

#DATA_NAME=virus
#N_SPLIT=10
#INCREMENTAL_START=10
#SUMMARY_THRESHOLD=70
#SELECTIVE_PRUNE_THRESHOLD=10 

#DATA_NAME=eukarya-lcc
#N_SPLIT=50
#INCREMENTAL_START=50 
#SUMMARY_THRESHOLD=70 
#SELECTIVE_PRUNE_THRESHOLD=10 
#INPUT_DIR=$CFS/m1982/taufique/eukarya-lcc-incremental/"$N_SPLIT"-split/
#INFILE_PREFIX=lcc_eukarya_30_50_length

DATA_NAME=eukarya
N_SPLIT=50
INCREMENTAL_START=50 
SUMMARY_THRESHOLD=70 
SELECTIVE_PRUNE_THRESHOLD=10 
INPUT_DIR=$CFS/m1982/taufique/eukarya-incremental/"$N_SPLIT"-split/
INFILE_PREFIX=eukarya_30_50_length

GT_DIR_NAME=incremental.$DATA_NAME.$N_SPLIT.$INCREMENTAL_START.$SUMMARY_THRESHOLD.$SELECTIVE_PRUNE_THRESHOLD.$SYSTEM.node_"$N_NODE".proc_"$PROC_PER_NODE".thread_"$THREAD_PER_PROC"

#for INCREMENTAL_START in 4 6 8 # For virus
for INCREMENTAL_START in 40 # For eukarya 20 30 40
do
    for SUMMARY_THRESHOLD in 90 70 50 30 # 30%, 50%, 70%, 90%
    do
        for SELECTIVE_PRUNE_THRESHOLD in 5 10 30 50 70 # 5% 10%, 30%, 50%, 70%
        do
            TEST_DIR_NAME=incremental.$DATA_NAME.$N_SPLIT.$INCREMENTAL_START.$SUMMARY_THRESHOLD.$SELECTIVE_PRUNE_THRESHOLD.$SYSTEM.node_"$N_NODE".proc_"$PROC_PER_NODE".thread_"$THREAD_PER_PROC"
            #TEST_DIR=$SCRATCH/$TEST_DIR_NAME/

            INCREMENTAL_END=$(( $N_SPLIT-1 ))
            for STEP in $(seq $INCREMENTAL_START $INCREMENTAL_END)
            do
                CLUSTER_FILE_GT=$SCRATCH/$GT_DIR_NAME/$STEP.inc
                CLUSTER_FILE_TEST=$SCRATCH/$TEST_DIR_NAME/$STEP.inc
                FSCORE_FILE=$SCRATCH/$TEST_DIR_NAME/$STEP.fscore
                echo GT: $CLUSTER_FILE_GT
                echo TEST: $CLUSTER_FILE_TEST

                if [ -f $FSCORE_FILE ]; then
                    echo "File $FSCORE_FILE already exists"
                    echo "Skipping"
                else
                    if [[ -f $CLUSTER_FILE_GT && -f $CLUSTER_FILE_TEST ]]; then 
                        $BINARY -M1 $CLUSTER_FILE_GT -M2 $CLUSTER_FILE_TEST -base 0 &> $FSCORE_FILE
                    else
                        echo "File $CLUSTER_FILE_GT does not exists"
                        echo "Or"
                        echo "File $CLUSTER_FILE_TEST does not exists"
                    fi
                fi
                echo ---
            done
        done
    done
done

