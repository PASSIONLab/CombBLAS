export OMP_NUM_THREADS=8
BINARY=/u/mth/Codes/CombBLAS/_build/Applications/Incremental/prep-data
IN_FILE=/home/mth/Data/nersc/geom/geom.mtx
OUT_PREFIX=/home/mth/Data/nersc/geom-90-10-split/geom

mpirun -n 4 $BINARY -I mm -base 1 -M $IN_FILE -old 0.9 -prefix $OUT_PREFIX
