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

M11FILE=/global/cscratch1/sd/taufique/eukarya-90-10-split/Renamed_euk_vs_euk_30_50length.indexed.m11.mtx
M12FILE=/global/cscratch1/sd/taufique/eukarya-90-10-split/Renamed_euk_vs_euk_30_50length.indexed.m12.mtx
M21FILE=/global/cscratch1/sd/taufique/eukarya-90-10-split/Renamed_euk_vs_euk_30_50length.indexed.m21.mtx
M22FILE=/global/cscratch1/sd/taufique/eukarya-90-10-split/Renamed_euk_vs_euk_30_50length.indexed.m22.mtx
MS11FILE=/global/cscratch1/sd/taufique/eukarya-90-10-split/Renamed_euk_vs_euk_30_50length.indexed.m11.mtx.summary
L11FILE=/global/cscratch1/sd/taufique/eukarya-90-10-split/Renamed_euk_vs_euk_30_50length.indexed.m11.lbl
L22FILE=/global/cscratch1/sd/taufique/eukarya-90-10-split/Renamed_euk_vs_euk_30_50length.indexed.m22.lbl
MSOFILE=/global/cscratch1/sd/taufique/eukarya-90-10-split/Renamed_euk_vs_euk_30_50length.indexed.mtx.summary
LOFILE=/global/cscratch1/sd/taufique/eukarya-90-10-split/Renamed_euk_vs_euk_30_50length.indexed.lbl
COFILE=/global/cscratch1/sd/taufique/eukarya-90-10-split/Renamed_euk_vs_euk_30_50length.indexed.inc
INC_BIN=/global/homes/t/taufique/Codes/CombBLAS/_build/Applications/Incremental/inc
srun -N 16 -n 64 -c 16 --ntasks-per-node=4 --cpu-bind=cores $INC_BIN -I mm -base 1 \
    -M11 $M11FILE \
    -M12 $M12FILE \
    -M21 $M21FILE \
    -M22 $M22FILE \
    -summary-in $MS11FILE \
    -L11 $L11FILE \
    -L22 $L22FILE \
    -label-out $LOFILE \
    -summary-out $MSOFILE \
    -cluster-out $COFILE $ \
    -inc v2 \
    -per-process-mem 20 &> debug.out.inc.euk.n16.s50

#srun -N 256 -n 1024 -c 16 --ntasks-per-node=4 --cpu-bind=cores $INC_BIN -I mm -M $MFILE -N 50 --per-process-mem 20 &> debug.out.$ALG.euk.n256.s50
