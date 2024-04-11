#!/bin/bash -l

#SBATCH -q debug 
#SBATCH -C cpu
##SBATCH -A m1982 # Aydin's project (PASSION: Scalable Graph Learning for Scientific Discovery)
##SBATCH -A m2865 # Exabiome project
##SBATCH -A m4012 # Ariful's project (GraphML: Intelligent Premitives for Graph Machine Learning)
#SBATCH -A m4293 # Sparsitute project (A Mathematical Institute for Sparse Computations in Science and Engineering)

#SBATCH -t 0:30:00

#SBATCH -N 1
#SBATCH -J mtx2petsc
#SBATCH -o slurm.mtx2petsc.o%j

SYSTEM=perlmutter_cpu
CORE_PER_NODE=128 # Never change. Specific to the system
PER_NODE_MEMORY=256 # Never change. Specific to the system
N_NODE=1
PROC_PER_NODE=8
N_PROC=$(( $N_NODE * $PROC_PER_NODE ))
THREAD_PER_PROC=$(( $CORE_PER_NODE / $PROC_PER_NODE ))
PER_PROC_MEM=$(( $PER_NODE_MEMORY / $PROC_PER_NODE - 2)) #2GB margin of error
export OMP_NUM_THREADS=$THREAD_PER_PROC

SCRIPT=$HOME/Codes/CombBLAS/Applications/Tall-Skinny-SpGEMM/scripts/ConvertMtxToPetsc.py 
STDOUT=out.mtx2petsc

conda activate env37

DATASET_HOME=$CFS/m4293/datasets
#for DATASET_NAME in uk-2002 it-2004 arabic-2005 GAP-web
#for DATASET_NAME in uk-2002
#for DATASET_NAME in it-2004 
#for DATASET_NAME in arabic-2005 
for DATASET_NAME in GAP-web
do
    A_FILE_MTX="$DATASET_HOME"/"$DATASET_NAME"/"$DATASET_NAME"."mtx"
    A_FILE_PSC="$DATASET_HOME"/"$DATASET_NAME"/"$DATASET_NAME"."petsc"
    
    if [ -f "$A_FILE_MTX" ]; then
        echo [x] "$A_FILE_MTX"
        if [ -f "$A_FILE_PSC" ]; then
            echo [x] "$A_FILE_PSC"
        else
            echo [.] "$A_FILE_PSC"
            ##srun -N $N_NODE -n $N_PROC python3 $SCRIPT --mtx $A_FILE_MTX --petsc $A_FILE_PSC >> $STDOUT
            #python3 $SCRIPT --mtx $A_FILE_MTX --petsc $A_FILE_PSC >> $STDOUT
        fi
    else
        echo [.] "$A_FILE_MTX"
    fi


    #for SP in 99.9 99 80
    for SP in 99
    do
        SPD_NAME=sp_"$SP"
        #for D in 4 16 64 128 256 1024 4096 16384 65536
        for D in 128
        do 
            DD_NAME=d_"$D"
            B_FILE_MTX="$DATASET_HOME"/"$DATASET_NAME"/"$SPD_NAME"/"$DD_NAME"/"sparse_local"."txt"
            B_FILE_PSC="$DATASET_HOME"/"$DATASET_NAME"/"$SPD_NAME"/"$DD_NAME"/"sparse_local"."petsc"

            if [ -f "$B_FILE_MTX" ]; then
                echo [x] "$B_FILE_MTX"
                if [ -f "$B_FILE_PSC" ]; then
                    echo [x] "$B_FILE_PSC"
                else
                    echo [.] "$B_FILE_PSC"
                    ##srun -N $N_NODE -n $N_PROC python3 $SCRIPT --mtx $B_FILE_MTX --petsc $B_FILE_PSC >> $STDOUT
                    python3 $SCRIPT --mtx $B_FILE_MTX --petsc $B_FILE_PSC >> $STDOUT
                fi
            else
                echo [.] "$B_FILE_MTX"
            fi


        done
    done

done

conda deactivate
