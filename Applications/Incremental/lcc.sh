#!/bin/bash -l

#SBATCH -q debug
#SBATCH -C knl
##SBATCH -A m1982 # Aydin's project (PASSION: Scalable Graph Learning for Scientific Discovery)
##SBATCH -A m2865 # Exabiome project
#SBATCH -A m2865 # Ariful's project (GraphML: Intelligent Premitives for Graph Machine Learning)

#SBATCH -t 00:30:00

#SBATCH -N 4
#SBATCH -J lcc
#SBATCH -o lcc.o%j

export OMP_NUM_THREADS=4

LCC_BIN=/global/homes/t/taufique/Codes/CombBLAS/_build/Applications/Incremental/lcc
PREP_SPLIT_BIN=/global/homes/t/taufique/Codes/CombBLAS/_build/Applications/Incremental/prep-split
LACC_BIN=/global/homes/t/taufique/Codes/CombBLAS/_build/Applications/lacc

MFILE=/global/cscratch1/sd/taufique/vir_vs_vir_30_50length_propermm.mtx
CFILE=/global/cscratch1/sd/taufique/vir_vs_vir_30_50length_propermm.mtx.components
#MFILE=/global/cscratch1/sd/taufique/Renamed_euk_vs_euk_30_50length.indexed.mtx
#CFILE=/global/cscratch1/sd/taufique/Renamed_euk_vs_euk_30_50length.indexed.mtx.components
#MFILE=/global/cscratch1/sd/taufique/Renamed_arch_vs_arch_30_50length.indexed.mtx
#CFILE=/global/cscratch1/sd/taufique/Renamed_arch_vs_arch_30_50length.indexed.mtx.components
#srun -N 4 -n 64 -c 4 --ntasks-per-node=16 --cpu-bind=cores valgrind --leak-check=yes --track-origins=yes --keep-stacktraces=alloc-and-free $BIN $IN_FILE
#srun -N 4 -n 64 -c 4 --ntasks-per-node=16 --cpu-bind=cores $LCC_BIN -I mm -M $MFILE -C $CFILE
#srun -N 4 -n 64 -c 4 --ntasks-per-node=16 --cpu-bind=cores $LACC_BIN -I mm -M $MFILE
srun -N 4 -n 64 -c 4 --ntasks-per-node=16 --cpu-bind=cores $PREP_SPLIT_BIN -I mm -M $MFILE -N 4
