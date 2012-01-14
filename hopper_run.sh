# in your script, don't forget the following line for any shared library
setenv CRAY_ROOTFS DSL

# Specifically for KDT, also include 
setenv PYTHONPATH ${SCRATCH}/lib/python
setenv LD_LIBRARY_PATH ${LD_LIBRARY_PATH}:${PYTHONPATH}

# These assume that your script is in C-Shell, meaning the first line is as follows:
#PBS -S /usr/bin/csh


# Full example (under /kdt/trunk/examples):

#PBS -S /usr/bin/csh
#PBS -q debug
#PBS -l mppwidth=529
#PBS -l walltime=00:30:00
#PBS -j eo
#PBS -V
setenv CRAY_ROOTFS DSL
setenv PYTHONPATH ${SCRATCH}/lib/python
setenv LD_LIBRARY_PATH ${LD_LIBRARY_PATH}:${PYTHONPATH}:${CRAY_LD_LIBRARY_PATH}
module unload xt-shmem
module load python

cd $PBS_O_WORKDIR
aprun -n 529 python BetwCent.py -g18 -x0.015 -b768 -d
