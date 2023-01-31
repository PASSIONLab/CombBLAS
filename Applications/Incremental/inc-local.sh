#!/bin/bash -l

export OMP_NUM_THREADS=4

#INC_BIN=/Users/mth/Codes/CombBLAS/_build/Applications/Incremental/inc-full
#INC_BIN=/Users/mth/Codes/CombBLAS/_build/Applications/Incremental/inc-base
INC_BIN=/Users/mth/Codes/CombBLAS/_build/Applications/Incremental/inc-opt
MCL_BIN=/Users/mth/Codes/CombBLAS/_build/Applications/mcl

#MFILE=/Users/mth/Data/cori/virus-lcc/vir_vs_vir_30_50length_propermm.mtx.lcc
#mpirun -n 4 $INC_BIN -I mm -M $MFILE -N 20 --per-process-mem 5
mpirun -n 4 $INC_BIN -I mm -M blah.txt -N 20 --per-process-mem 5

#mpirun -n 1 $MCL_BIN -I 2.0 -M blah.txt --matrix-market --per-process-mem 5 -o blah.mcl
