#!/bin/bash -l

#SBATCH -q regular
#SBATCH -C knl
#SBATCH -A m2865
#SBATCH -t 00:59:00

#SBATCH -N 1
#SBATCH -J spadd
#SBATCH -o spadd.o%j

##SBATCH -N 16
##SBATCH -J hipmcl_virus
##SBATCH -o hipmcl_virus.o%j

##SBATCH -N 256
##SBATCH -J cage15_cori_KNL_3D_[16x16x4]
##SBATCH -o cage15_cori_KNL_3D_[16x16x4].o%j

##SBATCH -N 32
##SBATCH -J hipmcl_eukarya
##SBATCH -o hipmcl_eukarya.o%j

##SBATCH -N 256
##SBATCH -J m100_subgraph3_cori_KNL_3D_[16x16x4]
##SBATCH -o m100_subgraph3_cori_KNL_3D_[16x16x4].o%j

##SBATCH -N 64
##SBATCH -J hipmcl_subgraph1
##SBATCH -o hipmcl_subgraph1.o%j

##SBATCH -N 1024
##SBATCH -J metaclust_cori_KNL_3D_[16x16x16]
##SBATCH -o metaclust_cori_KNL_3D_[16x16x16].o%j

MCL_EXE=./mcl3d
SpGEMM2D_EXE=./SpGEMM2D
SpAdd_EXE=./CFEstimate
export OMP_NUM_THREADS=16

#IN_FILE=/global/cscratch1/sd/taufique/eukarya_int_3d/r63_s10
export OMP_NUM_THREADS=64
srun -N 1 -n 1 -c 64 --ntasks-per-node=1 --cpu-bind=cores $SpAdd_EXE

#IN_FILE=/project/projectdirs/m1982/HipMCL/viruses/vir_vs_vir_30_50length.indexed.triples
#IN_FILE=/global/homes/t/taufique/Data/dummy.triples
#OUT_FILE=/global/cscratch1/sd/taufique/vir_vs_vir_30_50length.indexed.triples.mcl3d
#srun -N 4 -n 16 -c 16 --ntasks-per-node=4 --cpu-bind=cores $MCL_EXE -M $IN_FILE -I 2 -per-process-mem 27 -layers 1 -o $OUT_FILE
#srun -N 4 -n 16 -c 16 --ntasks-per-node=4 --cpu-bind=cores $MCL_EXE -M $IN_FILE -I 2 -per-process-mem 27 -layers 4 -o $OUT_FILE
##srun -N 1 -n 16 -c 4 --ntasks-per-node=16 --cpu-bind=cores valgrind --leak-check=full --show-reachable=yes --log-file=nc.vg.%p $MCL_EXE -M $IN_FILE -I 2 -per-process-mem 6 -layers 4 -o $OUT_FILE &> hipmcl_virus.oINTERACTIVE
#srun -N 16 -n 64 -c 16 --ntasks-per-node=4 --cpu-bind=cores $SpGEMM2D_EXE $IN_FILE

#IN_FILE=/project/projectdirs/m1982/HipMCL/eukarya/euk_vs_euk_30_50length.indexed.triples
##OUT_FILE=/global/cscratch1/sd/taufique/euk_vs_euk_30_50length.indexed.triples.hipmcl3d
####srun -N 16 -n 64 -c 16 --ntasks-per-node=4 --cpu-bind=cores $MCL_EXE -M $IN_FILE -I 2 -per-process-mem 27 -layers 1 -o $OUT_FILE
###srun -N 16 -n 64 -c 16 --ntasks-per-node=4 --cpu-bind=cores $MCL_EXE -M $IN_FILE -I 2 -per-process-mem 27 -layers 4 -o $OUT_FILE
#srun -N 32 -n 64 -c 16 --ntasks-per-node=2 --cpu-bind=cores $SpGEMM2D_EXE $IN_FILE

#IN_FILE=/project/projectdirs/m1982/HipMCL/iso_m100/subgraph3/subgraph3_iso_vs_iso_30_70length_ALL.m100.indexed.mtx
#OUT_FILE=/global/cscratch1/sd/taufique/subgraph3_iso_vs_iso_30_70length_ALL.m100.indexed.mtx.mcl3d
#srun -N 256 -n 1024 -c 16 --ntasks-per-node=4 --cpu-bind=cores $MCL_EXE -M $IN_FILE --matrix-market -base 0 -I 2 -per-process-mem 27 -layers 4 -o $OUT_FILE

#IN_FILE=/global/cscratch1/sd/taufique/subgraph1_iso_vs_iso_30_70length_ALL.m100.indexed.triples
#OUT_FILE=/global/cscratch1/sd/taufique/subgraph1_iso_vs_iso_30_70length_ALL.m100.indexed.triples.hipmcl3d
#echo "layer=16"
#srun -N 64 -n 256 -c 16 --ntasks-per-node=4 --cpu-bind=cores $MCL_EXE -M $IN_FILE -I 2 -per-process-mem 27 -layers 16 -o $OUT_FILE
#echo "layer=4"
#srun -N 64 -n 256 -c 16 --ntasks-per-node=4 --cpu-bind=cores $MCL_EXE -M $IN_FILE -I 2 -per-process-mem 27 -layers 4 -o $OUT_FILE
##srun -N 256 -n 1024 -c 16 --ntasks-per-node=4 --cpu-bind=cores $MCL_EXE -M $IN_FILE -I 2 -per-process-mem 27 -layers 16 -o $OUT_FILE
##srun -N 4096 -n 16384 -c 16 --ntasks-per-node=4 --cpu-bind=cores $MCL_EXE -M $IN_FILE -I 2 -per-process-mem 27 -layers 16 -o $OUT_FILE

#IN_FILE=/global/cscratch1/sd/taufique/Metaclust50_MATRIX_DENSE.txt_temp
#OUT_FILE=/global/cscratch1/sd/taufique/Metaclust50_MATRIX_DENSE.txt_temp.hipmcl.mcl3d
#srun -N 1024 -n 4096 -c 16 --ntasks-per-node=4 --cpu-bind=cores $MCL_EXE -M $IN_FILE -I 2 -per-process-mem 27 -layers 16 -o $OUT_FILE
