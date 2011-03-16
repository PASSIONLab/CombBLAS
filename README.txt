
Installation
------------

KDT is distributed as a Python Distutils package. It requires the MPI compiler
to be specified in the $CC and $CXX environment variables. For example, in Bash:
export CC=mpicxx
export CXX=mpicxx

The build and installation is performed by the standard Distutils setup command:
$ python setup.py build
$ sudo python setup.py install

If the build step fails fails see the "Choosing a compiler" section below.

System Requirements
-------------------

We recommend Python 2.4 or newer.

KDT makes use of some features from C++ TR1. If your compiler does not support
TR1 then the free Boost C++ library (http://www.boost.org/) supplies the
required headers as well. Make sure boost/ is in your include path. If it is
not, you can append the include path with the -I switch to the setup.py
script. For example, if you installed Boost in /home/username/include/boost:
$ python setup.py build -I/home/username/include

The MPI library must be compiled with -fPIC. Python modules must be compiled
with -fPIC, and that includes all libraries that get linked into the module.
That includes the MPI library in the case of KDT. If your MPI was not
compiled with -fPIC then the link step will fail. If this happens, contact your
system administrator.

Choosing a compiler
-------------------

KDT consists of pure Python classes and a C++ extension. This C++ extension,
called pyCombBLAS, is MPI code and must be compiled as such. However, it is
also a Python module, so must be compiled with a compiler that is compatible
with your Python installation. By default, distutils (i.e. setup.py) will use
the same compiler that Python was compiled with. That was probably not an MPI
compiler, so the proper compiler must be specified to distutils via the CC and
CXX environment variables.

Note that your system may give you a choice between GNU and Portland Group
(PGI) compilers. These are not fully binary compatible with each other, so you
must use the same one that your Python was compiled with. Note that mpicxx is
often just wrapper around GNU or PGI compilers.

System Libraries
----------------

If you chose to use a non-default MPI compiler then be aware that the runtime
may link to the default MPI libraries anyway. This is probably not what you
want. The result may be something like this:

ImportError: ./kdt/_pyCombBLAS.so: undefined symbol: _ZN3MPI3Win4FreeEv

The solution is to set your LD_LIBRARY environment variable such that the MPI
library you compiled with appears before the incorrect defaults.


Building without Distutils
--------------------------

If the setup.py script fails for you, you can build the C++ extension
manually. The kdt/pyCombBLAS directory contains a Makefile (named
Makefile-dist, rename as necessary) which you can tune
for your system. The important things to change are:
COMPILER - compiler to use.
INCADD - include your Python build directory as well as your Boost
installation if you need it.
OPT - any flags you wish to change.

Once pyCombBLAS is built, you may manually install the kdt/ directory in your
systemwide Python site-packages directory, or simply set your PYTHONPATH environment
variable to point to the parent directory of kdt/ (i.e. the base of the
distribution).
