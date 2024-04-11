#!/bin/bash -l

#SBATCH -q regular 
#SBATCH -C cpu
##SBATCH -A m1982 # Aydin's project (PASSION: Scalable Graph Learning for Scientific Discovery)
##SBATCH -A m2865 # Exabiome project
#SBATCH -A m4012 # Ariful's project (GraphML: Intelligent Premitives for Graph Machine Learning)
##SBATCH -A m4293 # Sparsitute project (A Mathematical Institute for Sparse Computations in Science and Engineering)

#SBATCH -t 2:30:00

#SBATCH -N 1
#SBATCH -J p4-ss
#SBATCH -o slurm.p4-ss.o%j

SYSTEM=perlmutter_cpu
CORE_PER_NODE=128 # Never change. Specific to the system
PER_NODE_MEMORY=256 # Never change. Specific to the system
N_NODE=1
PROC_PER_NODE=8
#N_PROC=$(( $N_NODE * $PROC_PER_NODE ))
N_PROC=4
#THREAD_PER_PROC=$(( $CORE_PER_NODE / $PROC_PER_NODE ))
THREAD_PER_PROC=16
PER_PROC_MEM=$(( $PER_NODE_MEMORY / $PROC_PER_NODE - 2)) #2GB margin of error
export OMP_NUM_THREADS=$THREAD_PER_PROC

PSC_BINARY=$HOME/Codes/CombBLAS/_build/Applications/Tall-Skinny-SpGEMM/petsc_test
CB_BINARY=$HOME/Codes/CombBLAS/_build/Applications/Tall-Skinny-SpGEMM/summa_test

DATASET_HOME=$CFS/m4293/datasets
#for DATASET_NAME in uk-2002 it-2004 arabic-2005 GAP-web
#for DATASET_NAME in uk-2002 it-2004
#for DATASET_NAME in GAP-web
for DATASET_NAME in arabic-2005
#for DATASET_NAME in it-2004
#for DATASET_NAME in uk-2002
do

    A_FILE_MTX="$DATASET_HOME"/"$DATASET_NAME"/"$DATASET_NAME"."mtx"
    A_FILE_PSC="$DATASET_HOME"/"$DATASET_NAME"/"$DATASET_NAME"."petsc"

    if [ -f "$A_FILE_PSC" ]; then
        echo [x] "$A_FILE_PSC"
    else
        echo [.] "$A_FILE_PSC"
    fi

    if [ -f "$A_FILE_MTX" ]; then
        echo [x] "$A_FILE_MTX"
    else
        echo [.] "$A_FILE_MTX"
    fi

    #for SP in 80 99 99.9
    for SP in 80 99
    do
        SPD_NAME=sp_"$SP"
        #for D in 4 16 64 128 256 1024 4096 16384 65536
        for D in 128
        do 
            DD_NAME=d_"$D"
            B_FILE_MTX="$DATASET_HOME"/"$DATASET_NAME"/"$SPD_NAME"/"$DD_NAME"/"sparse_local"."txt"
            B_FILE_PSC="$DATASET_HOME"/"$DATASET_NAME"/"$SPD_NAME"/"$DD_NAME"/"sparse_local"."petsc"
            C_FILE_MTX="$DATASET_HOME"/"$DATASET_NAME"/"$SPD_NAME"/"$DD_NAME"/"c"."txt"
            C_FILE_PSC="$DATASET_HOME"/"$DATASET_NAME"/"$SPD_NAME"/"$DD_NAME"/"c"."petsc"

            if [ -f "$B_FILE_MTX" ]; then
                echo [x] "$B_FILE_MTX"
                # Inputs ready to run SUMMA 2D and 3D
                OUT_FILE="$DATASET_HOME"/"$DATASET_NAME"/"$SPD_NAME"/"$DD_NAME"/summa2d.p"$N_PROC".n"$N_NODE"
                if [ -f "$OUT_FILE" ]; then
                    # Previously been run, result is ready
                    echo [x] "$OUT_FILE"
                else
                    ## run summa2d
                    echo "Running SUMMMA-2D"
                    srun -N $N_NODE -n $N_PROC -c $THREAD_PER_PROC --ntasks-per-node=$PROC_PER_NODE --cpu-bind=cores $CB_BINARY -A $A_FILE_MTX -B $B_FILE_MTX -permute sym -layer 1 &> "$OUT_FILE"
                fi

                OUT_FILE="$DATASET_HOME"/"$DATASET_NAME"/"$SPD_NAME"/"$DD_NAME"/summa3d.p"$N_PROC".n"$N_NODE"
                if [ -f "$OUT_FILE" ]; then
                    # Previously been run, result is ready
                    echo [x] "$OUT_FILE"
                else
                    ## run summa3d
                    echo "Running SUMMMA-3D"
                    N_LAYER=4
                    if [ "$N_PROC" == "4096" ]; then
                        N_LAYER=16
                    fi
                    srun -N $N_NODE -n $N_PROC -c $THREAD_PER_PROC --ntasks-per-node=$PROC_PER_NODE --cpu-bind=cores $CB_BINARY -A $A_FILE_MTX -B $B_FILE_MTX -permute sym -layer $N_LAYER &> "$OUT_FILE"
                fi
            else
                echo [.] "$B_FILE_MTX"
            fi

            if [ -f "$B_FILE_PSC" ]; then
                echo [x] "$B_FILE_PSC"
                OUT_FILE="$DATASET_HOME"/"$DATASET_NAME"/"$SPD_NAME"/"$DD_NAME"/petsc.p"$N_PROC".n"$N_NODE"
                if [ -f "$OUT_FILE" ]; then
                    # Previously been run, result is ready
                    echo [x] "$OUT_FILE"
                else
                    ## run summa2d
                    echo "Running PETSc"
                    srun -N $N_NODE -n $N_PROC -c $THREAD_PER_PROC --ntasks-per-node=$PROC_PER_NODE --cpu-bind=cores $PSC_BINARY $A_FILE_PSC $B_FILE_PSC $C_FILE_PSC 1 &> "$OUT_FILE"
                fi
            else
                echo [.] "$B_FILE_PSC"
            fi


        done
    done
done
