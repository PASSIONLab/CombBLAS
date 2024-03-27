#!/bin/bash -l

SCRIPT=$HOME/Codes/CombBLAS/Applications/Tall-Skinny-SpGEMM/scripts/ConvertMtxToPetsc.py 
STDOUT=out.mtx2petsc

conda activate env37

DATASET_HOME=$CFS/m4293/datasets
for DATASET_NAME in uk-2002 it-2004 arabic-2005 GAP-web
#for DATASET_NAME in it-2004 
#for DATASET_NAME in arabic-2005 
#for DATASET_NAME in GAP-web
do
    A_FILE_MTX="$DATASET_HOME"/"$DATASET_NAME"/"$DATASET_NAME"."mtx"
    A_FILE_PSC="$DATASET_HOME"/"$DATASET_NAME"/"$DATASET_NAME"."petsc"
    B_FILE_MTX="$DATASET_HOME"/"$DATASET_NAME"/"sp_80"/"d_128"/"sparse_local"."txt"
    B_FILE_PSC="$DATASET_HOME"/"$DATASET_NAME"/"sp_80"/"d_128"/"sparse_local"."petsc"

    if [ -f "$A_FILE_PSC" ]; then
        echo "$A_FILE_PSC" exists
    else
        python3 $SCRIPT --mtx $A_FILE_MTX --petsc $A_FILE_PSC >> $STDOUT
    fi

    if [ -f "$B_FILE_PSC" ]; then
        echo "$B_FILE_PSC" exists
    else
        python3 $SCRIPT --mtx $B_FILE_MTX --petsc $B_FILE_PSC >> $STDOUT
    fi
done

conda deactivate
