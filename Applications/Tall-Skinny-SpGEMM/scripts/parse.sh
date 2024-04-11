#!/bin/bash -l

conda activate env37

SCRIPT=$HOME/Codes/CombBLAS/Applications/Tall-Skinny-SpGEMM/scripts/parse.py 
CSV_FNAME=$CFS/m4293/datasets/baselines.csv
if [ -f $CSV_FILE ]; then
    mv $CSV_FILE $CSV_FILE.backup
    rm $CSV_FILE
fi
echo "data,n,p,d,s,impl,comm,comp,tot,abcast,bbcast,local_mult,layer_merge,fiber_reduct,fiber_merge" > $CSV_FNAME

DATASET_HOME=$CFS/m4293/datasets
for DATASET_NAME in uk-2002 it-2004 arabic-2005 GAP-web
#for DATASET_NAME in uk-2002
do
    for SP in 80 99 99.9
    #for SP in 80
    do
        SPD_NAME=sp_"$SP"
        for D in 4 16 64 128 256 1024 4096 16384 65536
        #for D in 128
        do 
            DD_NAME=d_"$D"
            B_FILE_MTX="$DATASET_HOME"/"$DATASET_NAME"/"$SPD_NAME"/"$DD_NAME"/"sparse_local"."txt"
            B_FILE_PSC="$DATASET_HOME"/"$DATASET_NAME"/"$SPD_NAME"/"$DD_NAME"/"sparse_local"."petsc"

            if [ -f $B_FILE_MTX ]; then
                for ALG in petsc summa2d summa3d
                #for ALG in summa3d
                do
                    for N_NODE in 1 2 8 32 128 512
                    #for N_NODE in 8
                    do
                        PROC_PER_NODE=8
                        N_PROC=$(( $N_NODE * $PROC_PER_NODE ))
                        LOG_FNAME=$DATASET_HOME/$DATASET_NAME/"$SPD_NAME"/"$DD_NAME"/"$ALG".p"$N_PROC".n"$N_NODE"
                        #CSV_FNAME=$DATASET_HOME/$DATASET_NAME/"$SPD_NAME"/"$DD_NAME"/"$ALG".p"$N_PROC".n"$N_NODE".csv

                        if [ "$N_NODE" == "1" ]; then
                            LOG_FNAME=$DATASET_HOME/$DATASET_NAME/"$SPD_NAME"/"$DD_NAME"/"$ALG".p4.n1
                        fi

                        if [ -f $LOG_FNAME ]; then
                            python3 $SCRIPT \
                                --data $DATASET_NAME \
                                --n $N_NODE \
                                --p $N_PROC \
                                --d $D \
                                --s $SP \
                                --alg $ALG \
                                --logf $LOG_FNAME \
                                --csvf $CSV_FNAME >> $CSV_FNAME

                        else
                            echo [.] $LOG_FNAME
                        fi

                    done
                done
            fi
        done
    done
done

conda deactivate
