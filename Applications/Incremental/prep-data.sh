#!/bin/bash -l

#SBATCH -q debug 
#SBATCH -C knl
##SBATCH -A m1982 # Aydin's project (PASSION: Scalable Graph Learning for Scientific Discovery)
##SBATCH -A m2865 # Exabiome project
#SBATCH -A m4012 # Ariful's project (GraphML: Intelligent Premitives for Graph Machine Learning)

#SBATCH -t 0:30:00

#SBATCH -N 64
#SBATCH -J prep-data
#SBATCH -o slurm.prep-data.o%j

export OMP_NUM_THREADS=16
N_NODE=16
PROC_PER_NODE=4
N_PROC=64
THREAD_PER_PROC=16

#BINARY=/global/homes/t/taufique/Codes/CombBLAS/_build/Applications/Incremental/prep-data
#IN_FILE=/global/cscratch1/sd/taufique/virus/vir_vs_vir_30_50length_propermm.mtx
#OUT_PREFIX=/global/cscratch1/sd/taufique/virus-incremental/20-split/virus_30_50
#N_SPLIT=20
#srun -N $N_NODE -n $N_PROC -c $THREAD_PER_PROC --ntasks-per-node=$PROC_PER_NODE --cpu-bind=cores \
    #$BINARY -I mm -M $IN_FILE -N $N_SPLIT -out-prefix $OUT_PREFIX &> out.prep-data.virus

BINARY=/global/homes/t/taufique/Codes/CombBLAS/_build/Applications/Incremental/prep-data
IN_FILE=/global/cscratch1/sd/taufique/eukarya-debug/Renamed_euk_vs_euk_30_50length.indexed.mtx
OUT_PREFIX=/global/cscratch1/sd/taufique/eukarya-incremental/50-split/eukarya_30_50_length
N_SPLIT=50
START_SPLIT=41
END_SPLIT=50
srun -N $N_NODE -n $N_PROC -c $THREAD_PER_PROC --ntasks-per-node=$PROC_PER_NODE --cpu-bind=cores \
    $BINARY -I mm -M $IN_FILE -out-prefix $OUT_PREFIX \
    -num-split $N_SPLIT -split-start $START_SPLIT -split-end $END_SPLIT  &> out.prep-data.eukarya
