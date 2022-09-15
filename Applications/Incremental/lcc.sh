#!/bin/bash -l

#SBATCH -q debug
#SBATCH -C knl
##SBATCH -A m1982 # Aydin's project (PASSION: Scalable Graph Learning for Scientific Discovery)
##SBATCH -A m2865 # Exabiome project
#SBATCH -A m4012 # Ariful's project (GraphML: Intelligent Premitives for Graph Machine Learning)

#SBATCH -t 00:30:00

#SBATCH -N 256
#SBATCH -J lcc
#SBATCH -o lcc.o%j

export OMP_NUM_THREADS=4

LCC_BIN=/global/homes/t/taufique/Codes/CombBLAS/_build/Applications/Incremental/lcc

#MFILE=/global/cscratch1/sd/taufique/vir_vs_vir_30_50length_propermm.mtx
#srun -N 1 -n 16 -c 4 --ntasks-per-node=16 --cpu-bind=cores $PREP_SPLIT_BIN -I mm -M $MFILE -N 20

MFILE=/global/cscratch1/sd/taufique/Renamed_euk_vs_euk_30_50length.indexed.mtx
srun -N 256 -n 4096 -c 4 --ntasks-per-node=16 --cpu-bind=cores $LCC_BIN -I mm -M $MFILE

#MFILE=/global/cscratch1/sd/taufique/Renamed_arch_vs_arch_30_50length.indexed.mtx
#srun -N 4 -n 64 -c 4 --ntasks-per-node=16 --cpu-bind=cores $LCC_BIN -I mm -M $MFILE
