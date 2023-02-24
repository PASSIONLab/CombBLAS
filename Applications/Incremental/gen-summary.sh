export OMP_NUM_THREADS=4
BINARY=/u/mth/Codes/CombBLAS/_build/Applications/Incremental/testideas
IN_FILE=/home/mth/Data/nersc/geom-90-10-split/geom.m11.mtx

mpirun -n 4 $BINARY -I mm -base 1 -M $IN_FILE --per-process-mem 8
