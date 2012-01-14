#!/bin/bash -l
echo "Do not run this file as a script, execute its commands one by one; please. Otherwise, you won't get the exports correctly"
module unload xt-shmem
module swap PrgEnv-pgi PrgEnv-gnu
module load python
export CC="cc -shared -dynamic"
export CXX="CC -shared -dynamic"
export XTPE_LINK_TYPE=dynamic
echo $CXX
python setup.py build
python setup.py install --home=${SCRATCH}
export PYTHONPATH=${SCRATCH}/lib/python
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:${PYTHONPATH}
