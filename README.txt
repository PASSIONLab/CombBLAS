
Installation
------------

KDT is distributed as a Python Distutils package. It includes an MPI C++
extension that must be compiled using an MPI compiler (eg. mpicxx).
This should be autodetected for you, but you may specify your preferred
MPI compiler in mpi.cfg. This file includes examples for some common MPI
distributions. It is the same file that mpi4py uses.

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

Choosing a Compiler
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

Problems with System Libraries
------------------------------

If you chose to use a non-default MPI compiler but still use the default mpirun,
then you will still use the default MPI libraries. This is probably not what you
want. The result may be something like this:

ImportError: ./kdt/_pyCombBLAS.so: undefined symbol: _ZN3MPI3Win4FreeEv

The solution is to fully specify the path to mpirun. If that still does not
solve the issue, set your LD_LIBRARY environment variable such that the MPI
library you compiled with appears before the incorrect defaults.

Problems with OpenMPI
---------------------

Some versions of OpenMPI do not properly link to themselves. This can manifest
in undefined symbol errors when the kdt module is loaded.

The solution is to manually add the OpenMPI lib path to the library path before
running your Python script:
export LD_LIBRARY_PATH=/opt/openmpi/gnu/mx/lib:$LD_LIBRARY_PATH

Substitute the appropriate path for your system.
