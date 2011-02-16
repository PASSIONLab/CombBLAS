
Installation
------------

KDT is distributed as a Python Distutils package. It requires the MPI compiler to be specified in the $CC and $CXX environment variables. For example, in Bash:
export CC=mpicxx
export CXX=mpicxx

The build and installation is performed by the standard Distutils setup command:
$ python setup.py build
$ sudo python setup.py install

System Requirements
-------------------

KDT makes use of some features from C++ TR1. If your compiler does not support TR1 then the free Boost C++ library (http://www.boost.org/) supplies the required headers as well. Make sure boost/ is in your include path. If it is not, you can append the include path with the -I switch to the setup.py script. For example, if you installed Boost in /home/username/include/boost:
$ python setup.py build -I/home/username/include

MPI library must be compiled with -fPIC. Python modules must be compiled with -fPIC, and that includes all libraries that get linked into the module. That includes the MPI library in the case of KDT. If your MPI was not compiled with -fPIC then the link step will fail. If this happens, contact your system administrator.

