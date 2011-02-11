
Installation
------------

KDT is distributed as a Python Distutils package. It requires the MPI compiler to be specified in the $CC and $CXX environment variables. For example, in Bash:
export CC=mpicxx
export CXX=mpicxx

The build and installation is performed by the standard Distutils setup command:
$ python setup.py build
$ sudo python setup.py install

Note that KDT makes use of some features from TR1. If your compiler does not support TR1 then the free Boost C++ library (http://www.boost.org/) supplies the required headers as well. Make sure boost/ is in your include path.

