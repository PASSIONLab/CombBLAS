#!/bin/bash -l

DATASET_HOME=$CFS/m4293/datasets
for DATASET_NAME in uk-2002 it-2004 arabic-2005 GAP-web
#for DATASET_NAME in it-2004
do
    #for SP in 80 99 99.9
    for SP in 80 99
    #for SP in 80
    do
        SPD_NAME=sp_"$SP"
        #for D in 4 16 64 128 256 1024 4096 16384 65536
        for D in 128
        do 
            DD_NAME=d_"$D"
            B_FILE_MTX="$DATASET_HOME"/"$DATASET_NAME"/"$SPD_NAME"/"$DD_NAME"/"sparse_local"."txt"
            B_FILE_PSC="$DATASET_HOME"/"$DATASET_NAME"/"$SPD_NAME"/"$DD_NAME"/"sparse_local"."petsc"

            if [ -f $B_FILE_MTX ]; then
                for ALG in petsc summa2d summa3d
                #for ALG in summa2d summa3d
                do
                    #for N_NODE in 8 32 128
                    for N_NODE in 2
                    do
                        PROC_PER_NODE=8
                        N_PROC=$(( $N_NODE * $PROC_PER_NODE ))
                        FNAME=$DATASET_HOME/$DATASET_NAME/"$SPD_NAME"/"$DD_NAME"/"$ALG".p"$N_PROC".n"$N_NODE"

                        #TFNAME=stdout.ss.$ALG.$DATASET_NAME.p$N_PROC
                        #mv $TFNAME $FNAME
                        if [ -f $FNAME ]; then
                            echo [x] $FNAME
                        else
                            echo [.] $FNAME
                        fi

                    done
                done
            fi
        done
    done
done
