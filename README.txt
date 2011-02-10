
Installation
------------

KDT is distributed as a Python Distutils package. It does require the MPI compiler to be specified as an environment variable. For example, in Bash:
export CC=mpicxx
export CXX=mpicxx

Then the standard Distutils setup command can be used:
$ python setup.py build
$ sudo python setup.py install

Note that KDT makes use of some features from TR1. If your compiler does not support TR1 then the free Boost library includes the required headers as well.

