#!/bin/bash -l

export OMP_NUM_THREADS=16

FSCORE_BIN=/global/homes/t/taufique/Codes/CombBLAS/Applications/mcl-runs/fscore

#MFILE=/global/cscratch1/sd/taufique/virus/vir_vs_vir_30_50length_propermm.mtx
#MFILE=/global/cscratch1/sd/taufique/virus-lcc/vir_vs_vir_30_50length_propermm.mtx.lcc
MFILE=/global/cscratch1/sd/taufique/eukarya/Renamed_euk_vs_euk_30_50length.indexed.mtx
#MFILE=/global/cscratch1/sd/taufique/email-Eu-core/email-Eu-core.mtx

NSPLIT=20
START_STEP=0
END_STEP=$(($NSPLIT - 1))
ALG_TEST=inc-opt-shuffled
ALG_GT=full
for STEP in $(seq $START_STEP $END_STEP);
    do echo $STEP; 
    CLUSTER_FILE_1=$MFILE.$NSPLIT.$ALG_TEST.$STEP
    CLUSTER_FILE_2=$MFILE.$NSPLIT.$ALG_GT.$STEP
    FSCORE_FILE=$CLUSTER_FILE_1.fscore
    echo $CLUSTER_FILE_1
    echo $CLUSTER_FILE_2
    #echo $FSCORE_BIN -M1 $CLUSTER_FILE_1 -M2 $CLUSTER_FILE_2 -base 0 &> $FSCORE_FILE
    $FSCORE_BIN -M1 $CLUSTER_FILE_1 -M2 $CLUSTER_FILE_2 -base 0 &> $FSCORE_FILE
    echo ---
done


