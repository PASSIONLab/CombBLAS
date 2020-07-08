#!/bin/bash -l

#SBATCH -q debug
#SBATCH -C knl
#SBATCH -A m2865
#SBATCH -t 00:30:00

##SBATCH -N 4
##SBATCH -J hipmcl_virus
##SBATCH -o hipmcl_virus.o%j

##SBATCH -N 256
##SBATCH -J cage15_cori_KNL_3D_[16x16x4]
##SBATCH -o cage15_cori_KNL_3D_[16x16x4].o%j

#SBATCH -N 16
#SBATCH -J eukarya_cori_KNL_3D_[16x16x4]
#SBATCH -o eukarya_cori_KNL_3D_[16x16x4].o%j

##SBATCH -N 256
##SBATCH -J m100_subgraph3_cori_KNL_3D_[16x16x4]
##SBATCH -o m100_subgraph3_cori_KNL_3D_[16x16x4].o%j

##SBATCH -N 1024
##SBATCH -J metaclust_cori_KNL_3D_[16x16x16]
##SBATCH -o metaclust_cori_KNL_3D_[16x16x16].o%j

MCL_EXE=./mcl3d
export OMP_NUM_THREADS=16

#IN_FILE=/global/cscratch1/sd/taufique/vir_vs_vir_30_50length.indexed.triples
#OUT_FILE=/global/cscratch1/sd/taufique/vir_vs_vir_30_50length.indexed.triples.mcl3d
#srun -N 4 -n 16 -c 16 --ntasks-per-node=4 --cpu-bind=cores $MCL_EXE -M $IN_FILE -I 2 -per-process-mem 27 -layers 1 -o $OUT_FILE
#srun -N 4 -n 16 -c 16 --ntasks-per-node=4 --cpu-bind=cores $MCL_EXE -M $IN_FILE -I 2 -per-process-mem 27 -layers 4 -o $OUT_FILE

IN_FILE=/global/cscratch1/sd/taufique/euk_vs_euk_30_50length.indexed.triples
OUT_FILE=/global/cscratch1/sd/taufique/euk_vs_euk_30_50length.indexed.triples.mcl3d3d
srun -N 16 -n 64 -c 16 --ntasks-per-node=4 --cpu-bind=cores $MCL_EXE -M $IN_FILE -I 2 -per-process-mem 27 -layers 1 -o $OUT_FILE
srun -N 16 -n 64 -c 16 --ntasks-per-node=4 --cpu-bind=cores $MCL_EXE -M $IN_FILE -I 2 -per-process-mem 27 -layers 4 -o $OUT_FILE

#IN_FILE=/project/projectdirs/m1982/HipMCL/iso_m100/subgraph3/subgraph3_iso_vs_iso_30_70length_ALL.m100.indexed.mtx
#OUT_FILE=/global/cscratch1/sd/taufique/subgraph3_iso_vs_iso_30_70length_ALL.m100.indexed.mtx.mcl3d
#srun -N 256 -n 1024 -c 16 --ntasks-per-node=4 --cpu-bind=cores $MCL_EXE -M $IN_FILE --matrix-market -base 0 -I 2 -per-process-mem 27 -layers 4 -o $OUT_FILE

#IN_FILE=/global/cscratch1/sd/taufique/Metaclust50_MATRIX_DENSE.txt_temp
#OUT_FILE=/global/cscratch1/sd/taufique/Metaclust50_MATRIX_DENSE.txt_temp.hipmcl.mcl3d
#srun -N 1024 -n 4096 -c 16 --ntasks-per-node=4 --cpu-bind=cores $MCL_EXE -M $IN_FILE -I 2 -per-process-mem 27 -layers 16 -o $OUT_FILE
