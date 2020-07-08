#!/bin/bash -l

#SBATCH -C knl
#SBATCH -A m1982

##SBATCH -q debug
##SBATCH -t 00:30:00
##SBATCH -N 4
##SBATCH -J ss_eukarya_1to16l_4
##SBATCH -o ss_eukarya_1to16l_4.o%j
#IN_FILE=/global/cscratch1/sd/taufique/euk_vs_euk_30_50length.indexed.triples
#BINARY=./ss_triple_1to16l
#export OMP_NUM_THREADS=16
#srun -N 4 -n 16 -c 16 --ntasks-per-node=4 --cpu-bind=cores $BINARY $IN_FILE

##SBATCH -N 256
##SBATCH -J zz_cage15_cori
##SBATCH -o zz_cage15_cori.o%j
#IN_FILE=/global/cscratch1/sd/taufique/cage15.mtx

##SBATCH -q debug
##SBATCH -t 00:30:00
##SBATCH -N 256
##SBATCH -J ss_friendster_16l_256_symbolic
##SBATCH -o ss_friendster_16l_256_symbolic.o%j
#IN_FILE=/global/cscratch1/sd/taufique/com-Friendster/com-Friendster.mtx
#BINARY=./SpGEMM3D
#export OMP_NUM_THREADS=16
#srun -N 256 -n 1024 -c 16 --ntasks-per-node=4 --cpu-bind=cores $BINARY $IN_FILE

##SBATCH -q debug
##SBATCH -t 00:30:00
##SBATCH -N 256
##SBATCH -J zz_kmer_cori
##SBATCH -o zz_kmer_cori.o%j
#IN_FILE=/global/cscratch1/sd/taufique/kmer_V1r/kmer_V1r.mtx
##BINARY=./SpGEMM3D_Kmer_256
#BINARY=./SpGEMM3D1
#export OMP_NUM_THREADS=16
#srun -N 256 -n 1024 -c 16 --ntasks-per-node=4 --cpu-bind=cores $BINARY $IN_FILE

##SBATCH -q debug
##SBATCH -t 00:30:00
##SBATCH -N 256
##SBATCH -J zz_mawi_cori
##SBATCH -o zz_mawi_cori.o%j
#IN_FILE=/global/cscratch1/sd/taufique/mawi_201512020330/mawi_201512020330.mtx
##BINARY=./SpGEMM3D_Mawi_256
#BINARY=./SpGEMM3D
#export OMP_NUM_THREADS=16
#srun -N 256 -n 1024 -c 16 --ntasks-per-node=4 --cpu-bind=cores $BINARY $IN_FILE

##SBATCH -q debug
##SBATCH -t 00:30:00
##SBATCH -N 256
##SBATCH -J zz_twitter_cori
##SBATCH -o zz_twitter_cori.o%j
#IN_FILE=/global/cscratch1/sd/taufique/GAP-twitter/GAP-twitter.mtx
#BINARY=./SpGEMM3D1
#export OMP_NUM_THREADS=16
#srun -N 256 -n 1024 -c 16 --ntasks-per-node=4 --cpu-bind=cores $BINARY $IN_FILE

#SBATCH -q regular
#SBATCH -t 01:30:00
#SBATCH -N 64
#SBATCH -J ss_subgraph1_knl_symbolic_1to16l_64
#SBATCH -o ss_subgraph1_knl_symbolic_1to16l_64.o%j
#IN_FILE=/global/cscratch1/sd/taufique/subgraph1_iso_vs_iso_30_70length_ALL.m100.indexed.mtx
IN_FILE=/global/cscratch1/sd/taufique/subgraph1_iso_vs_iso_30_70length_ALL.m100.indexed.triples
BINARY=./symbolic_1to16l
export OMP_NUM_THREADS=16
srun -N 64 -n 256 -c 16 --ntasks-per-node=4 --cpu-bind=cores $BINARY $IN_FILE

##SBATCH -q regular
##SBATCH -t 01:00:00
##SBATCH -N 4096
##SBATCH -J ss_metaclust_16l_64l_4096_hyperthread
##SBATCH -o ss_metaclust_16l_64l_4096_hyperthread.o%j
##IN_FILE=/global/cscratch1/sd/azad/Renamed_graph_Metaclust50_MATRIX_DENSE.txt.mtx
#IN_FILE=/global/cscratch1/sd/azad/Renamed_graph_Metaclust50_MATRIX_DENSE.txt.nohead
#BINARY=./ss_metaclust_16l_64l_hyperthread
#export OMP_NUM_THREADS=16
##srun -N 256 -n 1024 -c 16 --ntasks-per-node=4 --cpu-bind=cores $BINARY $IN_FILE
#srun -N 4096 -n 65536 -c 16 --ntasks-per-node=16 --cpu-bind=threads $BINARY $IN_FILE

##SBATCH -q regular
##SBATCH -t 05:00:00
##SBATCH -N 256
##SBATCH -J ss_isolate_cori_256
##SBATCH -o ss_isolate_cori_256.o%j
#IN_FILE=/global/cscratch1/sd/taufique/iso_vs_iso_30_70length_ALL.m100.indexed.txt
#BINARY=./ss_triple_256
#export OMP_NUM_THREADS=16
#srun -N 256 -n 1024 -c 16 --ntasks-per-node=4 --cpu-bind=cores $BINARY $IN_FILE

##SBATCH -N 16
##SBATCH -J zz_webgoogle_cori
##SBATCH -o zz_webgoogle_cori.o%j
#IN_FILE=/global/cscratch1/sd/taufique/web-Google/web-Google.mtx
#BINARY=./SpGEMM3D
#export OMP_NUM_THREADS=16
#srun -N 16 -n 64 -c 16 --ntasks-per-node=4 --cpu-bind=cores $BINARY $IN_FILE

##SBATCH -q debug
##SBATCH -t 00:30:00
##SBATCH -N 16
##SBATCH -J zz_abaumannii_cori
##SBATCH -o zz_abaumannii_cori.o%j
#BINARY=./SpGEMM3D
#IN_FILE=/global/cscratch1/sd/taufique/abaumannii30x_readskmers.mtx
#export OMP_NUM_THREADS=16
#srun -N 16 -n 64 -c 16 --ntasks-per-node=4 --cpu-bind=cores $BINARY $IN_FILE

##SBATCH -q debug
##SBATCH -t 00:30:00
##SBATCH -N 64
##SBATCH -J zz_celegans_cori
##SBATCH -o zz_celegans_cori.o%j
#BINARY=./SpGEMM3D
#IN_FILE=/global/cscratch1/sd/taufique/celegans40x_readbykmers.mtx
#export OMP_NUM_THREADS=16
#srun -N 64 -n 256 -c 16 --ntasks-per-node=4 --cpu-bind=cores $BINARY $IN_FILE

##SBATCH -q debug
##SBATCH -t 00:30:00
##SBATCH -N 16
##SBATCH -J zz_srr_cori_16
##SBATCH -o zz_srr_cori_16.o%j
##BINARY=./SpGEMM3D
##BINARY=./ss_16l_symbolic
##IN_FILE=/global/cscratch1/sd/taufique/com-Friendster/com-Friendster.mtx
##srun -N 1024 -n 4096 -c 16 --ntasks-per-node=4 --cpu-bind=cores $BINARY $IN_FILE
##IN_FILE=/global/cscratch1/sd/taufique/subgraph1_iso_vs_iso_30_70length_ALL.m100.indexed.mtx
##srun -N 1024 -n 4096 -c 16 --ntasks-per-node=4 --cpu-bind=cores $BINARY $IN_FILE
##BINARY=./ss_rect_16l_1024
#BINARY=./SpGEMM3D
#IN_FILE=/global/cscratch1/sd/taufique/SRR3743363_readskmers2to4.mtx
#export OMP_NUM_THREADS=16
#srun -N 16 -n 64 -c 16 --ntasks-per-node=4 --cpu-bind=cores $BINARY $IN_FILE
