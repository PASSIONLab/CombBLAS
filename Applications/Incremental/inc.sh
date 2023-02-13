#!/bin/bash -l

#SBATCH -q debug 
#SBATCH -C knl
##SBATCH -A m1982 # Aydin's project (PASSION: Scalable Graph Learning for Scientific Discovery)
#SBATCH -A m2865 # Exabiome project
##SBATCH -A m4012 # Ariful's project (GraphML: Intelligent Premitives for Graph Machine Learning)

#SBATCH -t 00:30:00

#SBATCH -N 16
#SBATCH -J inc
##SBATCH -o out.inc.o%j

export OMP_NUM_THREADS=16

ALG=inc-v1
MCL_BIN=/global/homes/t/taufique/Codes/CombBLAS/_build/Applications/mcl
INC_BIN=/global/homes/t/taufique/Codes/CombBLAS/_build/Applications/Incremental/$ALG
#INC_BIN=/global/homes/t/taufique/Codes/CombBLAS/_build/Applications/Incremental/inc-base
#INC_BIN=/global/homes/t/taufique/Codes/CombBLAS/_build/Applications/Incremental/inc-opt

#MFILE=/global/cscratch1/sd/taufique/virus-lcc/vir_vs_vir_30_50length_propermm.mtx.lcc
#srun -N 1 -n 4 -c 16 --ntasks-per-node=4 --cpu-bind=cores $INC_BIN -I mm -M $MFILE -N 20 --per-process-mem 20 &> out.$ALG.virus-lcc.n1.s20

#MFILE=/global/cscratch1/sd/taufique/virus/vir_vs_vir_30_50length_propermm.mtx
#ALG=inc-v1
#INC_BIN=/global/homes/t/taufique/Codes/CombBLAS/_build/Applications/Incremental/$ALG
#srun -N 1 -n 4 -c 16 --ntasks-per-node=4 --cpu-bind=cores $INC_BIN -I mm -M $MFILE -N 20 --per-process-mem 20 &> out.$ALG.virus.n1.s20
#ALG=inc-base
#INC_BIN=/global/homes/t/taufique/Codes/CombBLAS/_build/Applications/Incremental/$ALG
#srun -N 1 -n 4 -c 16 --ntasks-per-node=4 --cpu-bind=cores $INC_BIN -I mm -M $MFILE -N 20 --per-process-mem 20 &> out.$ALG.virus.n1.s20
#ALG=inc-full
#INC_BIN=/global/homes/t/taufique/Codes/CombBLAS/_build/Applications/Incremental/$ALG
#srun -N 1 -n 4 -c 16 --ntasks-per-node=4 --cpu-bind=cores $INC_BIN -I mm -M $MFILE -N 20 --per-process-mem 20 &> out.$ALG.virus.n1.s20
#MFILE=/global/cscratch1/sd/taufique/virus/vir_vs_vir_30_50length_propermm.mtx
#srun -N 1 -n 4 -c 16 --ntasks-per-node=4 --cpu-bind=cores $MCL_BIN -I 2 -M $MFILE --matrix-market -per-process-mem 20 -o $MFILE.hipmcl

MFILE=/global/cscratch1/sd/taufique/eukarya/Renamed_euk_vs_euk_30_50length.indexed.mtx
ALG=inc-v1
INC_BIN=/global/homes/t/taufique/Codes/CombBLAS/_build/Applications/Incremental/$ALG
srun -N 16 -n 64 -c 16 --ntasks-per-node=4 --cpu-bind=cores $INC_BIN -I mm -M $MFILE -N 50 --per-process-mem 20 &> debug.out.$ALG.euk.n16.s50
#ALG=inc-base
#INC_BIN=/global/homes/t/taufique/Codes/CombBLAS/_build/Applications/Incremental/$ALG
#srun -N 16 -n 64 -c 16 --ntasks-per-node=4 --cpu-bind=cores $INC_BIN -I mm -M $MFILE -N 50 --per-process-mem 20 &> debug.out.$ALG.euk.n16.s50
#ALG=inc-full
#INC_BIN=/global/homes/t/taufique/Codes/CombBLAS/_build/Applications/Incremental/$ALG
#srun -N 16 -n 64 -c 16 --ntasks-per-node=4 --cpu-bind=cores $INC_BIN -I mm -M $MFILE -N 50 --per-process-mem 20 &> debug.out.$ALG.euk.n16.s50

#MFILE=/global/cscratch1/sd/taufique/email-Eu-core/email-Eu-core.mtx
#srun -N 1 -n 1 -c 64 --ntasks-per-node=1 --cpu-bind=cores $INC_BIN -I mm -M $MFILE -N 20 --per-process-mem 20
