#!/bin/bash -l

#SBATCH -q debug
#SBATCH -C knl
#SBATCH -A m2865
#SBATCH -t 00:30:00

##SBATCH -N 64
##SBATCH -J zz_eukarya_cori
##SBATCH -o zz_eukarya_cori.o%j

##SBATCH -N 256
##SBATCH -J zz_cage15_cori
##SBATCH -o zz_cage15_cori.o%j

#SBATCH -N 256
#SBATCH -J zz_friendster_cori
#SBATCH -o zz_friendster_cori.o%j

#BINARY=./SpGEMM2D
BINARY=./SpGEMM3D
#BINARY=./BcastTest
#BINARY=./CFEstimate

export OMP_NUM_THREADS=16

#IN_FILE=/global/cscratch1/sd/taufique/euk_vs_euk_30_50length.indexed.triples
#srun -N 256 -n 4096 -c 4 --ntasks-per-node=16 --cpu-bind=cores $BINARY $IN_FILE

## Labelled triple
#IN_FILE=/global/cscratch1/sd/taufique/cage15.mtx
#srun -N 256 -n 4096 -c 4 --ntasks-per-node=16 --cpu-bind=cores $BINARY $IN_FILE

# Matrix market file
#IN_FILE=/global/cscratch1/sd/taufique/web-Google/web-Google.mtx
IN_FILE=/global/cscratch1/sd/taufique/com-Friendster/com-Friendster.mtx
srun -N 256 -n 1024 -c 16 --ntasks-per-node=4 --cpu-bind=cores $BINARY $IN_FILE
