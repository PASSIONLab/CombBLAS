#!/usr/bin/env python

#export CC=mpicxx
#export CXX=mpicxx

from distutils.core import setup, Extension
from distutils import sysconfig
import sys

### print "Remember to set your preferred MPI C++ compiler in the CC and CXX environment variables. For example, in Bash:"
### print "export CC=mpicxx"
### print "export CXX=mpicxx"
### print ""

### def see_if_compiles(program, include_dirs):
### 	""" Try to compile the passed in program and report if it compiles successfully or not. """
###	from distutils.ccompiler import new_compiler, CompileError
###	from shutil import rmtree
###	import tempfile
###	import os
###	
###	try:
###		tmpdir = tempfile.mkdtemp()
###	except AttributeError:
###		# Python 2.2 doesn't have mkdtemp().
###		tmpdir = "compile_check_tempdir"
###		try:
###			os.mkdir(tmpdir)
###		except OSError:
###			print "Can't create temporary directory. Aborting."
###			sys.exit()
###			
###	old = os.getcwd()
###	
###	os.chdir(tmpdir)
###	
###	# Try to include the header
###	f = open('compiletest.cpp', 'w')
###	f.write(program)
###	f.close()
###	try:
###		new_compiler().compile([f.name], include_dirs=include_dirs)
###		success = True
###	except CompileError:
###		success = False
###	
###	os.chdir(old)
###	rmtree(tmpdir)
###	return success

### def check_for_header(header, include_dirs):
###	"""Check for the existence of a header file by creating a small program which includes it and see if it compiles."""
###	program = "#include <%s>\n" % header
###	sys.stdout.write("Checking for <%s>... " % header)
###	success = see_if_compiles(program, include_dirs)
###	if (success):
###		sys.stdout.write("OK\n");
###	else:
###		sys.stdout.write("Not found\n");
###	return success

### def check_for_MPI_IN_PLACE(include_dirs):
###	""" Check for the existence of the MPI_IN_PLACE constant. """
###	
###	program = """
### #include <mpi.h>
###
### int main(int argc, const char** argv) {
###	void* buf = NULL;
###	MPI_Allreduce(MPI_IN_PLACE, buf, 10, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
###	return 0;
### }
###
### """
###	sys.stdout.write("Checking for MPI_IN_PLACE... ")
###	success = see_if_compiles(program, include_dirs)
###	if (success):
###		sys.stdout.write("OK\n");
###	else:
###		sys.stdout.write("Not found\n");
###	return success

### # parse out additional include dirs from the command line
### include_dirs = []
### copy_args=sys.argv[1:]
### for a in copy_args:
###	if a.startswith('-I'):
###		include_dirs.append(a[2:])
###		copy_args.remove(a)

### # see if the compiler has TR1
### hasTR1 = False
### hasBoost = False
### headerDefs = []
### print "Checking for TR1..."
### if (check_for_header("tr1/memory", include_dirs) and check_for_header("tr1/tuple", include_dirs)):
###	hasTR1 = True
### else:
###	# nope, see if boost is available
###	print "No TR1. Checking for Boost instead..."
###	if (check_for_header("boost/tr1/memory.hpp", include_dirs) and check_for_header("boost/tr1/tuple.hpp", include_dirs)):
###		hasBoost = True
###		headerDefs = [('NOTR1', '1')]
###	else:
###		# nope, then sorry
###		print "KDT uses features from C++ TR1. These are available from some compilers or through the Boost C++ library (www.boost.org)."
###		print "Please make sure Boost is in your system include path or append the include path with the -I switch."
###		print "For example, if you have Boost installed in /home/username/include/boost:"
###		print "$ python setup.py build -I/home/username/include"
###		sys.exit();

#if (not check_for_MPI_IN_PLACE(include_dirs)):
#	print "Please use a more recent MPI implementation."
#	print "If you system has multiple MPI implementations you can set your preferred MPI C++ compiler in the CC and CXX environment variables. For example, in Bash:"
#	print "export CC=mpicxx"
#	print "export CXX=mpicxx"
#	sys.exit();


COMBBLAS = "CombBLAS/"
PCB = "kdt/pyCombBLAS/"
GENERATOR = "CombBLAS/graph500-1.2/generator/"

extra_compile_args = []
extra_link_args = []
macros = []

extra_compile_args.append('/Od')   # no optimizations, override the default /Ox
extra_compile_args.append('/Zi')   # debugging info
extra_link_args.append('/debug')   # debugging info
macros.insert(0, ('NDEBUG', '1'))  # prevents from being linked against python2x_d.lib

compatibility_include_dirs = []
compatibility_include_dirs.append(COMBBLAS+"ms_inttypes") # VS2008 does not include <inttypes.h>
compatibility_include_dirs.append(COMBBLAS+"ms_sys")      # VS2008 does not include <sys/time.h>, we include a blank one because other people's code includes it but none of the functions are used.
macros.append(('NODRAND48', '1'))                         # VS2008 does not include drand48(), which is used in psort
macros.append(('NOMINMAX', '1'))                          # Windows defines min and max as macros, which wreaks havoc with functions named min and max, regardless of namespace
macros.append(('_SCL_SECURE_NO_WARNINGS', '1'))           # disables odd but annoyingly verbose checks, maybe these are legit, don't know.
macros.append(('inline', '__inline'))
#macros.append(('restrict', '__restrict'))

#files for the graph500 graph generator.
generator_files = [GENERATOR+"btrd_binomial_distribution.c", GENERATOR+"splittable_mrg.c", GENERATOR+"mrg_transitions.c", GENERATOR+"graph_generator.c", GENERATOR+"permutation_gen.c", GENERATOR+"make_graph.c", GENERATOR+"utils.c", GENERATOR+"scramble_edges.c"]

#pyCombBLAS extension which wraps the templated C++ Combinatorial BLAS library. 
pyCombBLAS_ext = Extension('kdt._pyCombBLAS',
	[PCB+"pyCombBLAS.cpp", PCB+"pyCombBLAS_wrap.cpp", PCB+"pyDenseParVec.cpp",
PCB+"pyObjDenseParVec.cpp", PCB+"pySpParVec.cpp", PCB+"pySpParMat.cpp", PCB+"pySpParMatBool.cpp", PCB+"pyOperations.cpp", COMBBLAS+"CommGrid.cpp", COMBBLAS+"MPIType.cpp", COMBBLAS+"MemoryPool.cpp"] + generator_files,
        extra_compile_args = extra_compile_args,
        extra_link_args = extra_link_args,
        define_macros = macros,
	include_dirs=["C:\Program Files\MPICH2\include"] + compatibility_include_dirs,
	library_dirs=["C:\Program Files\MPICH2\lib"],
	libraries=["mpi","cxx"])
	#define_macros=[('NDEBUG', '1'),('restrict', '__restrict__'),('GRAPH_GENERATOR_SEQ', '1')] + headerDefs)

setup(name='kdt',
	version='0.1',
	description='Knowledge Discovery Toolbox',
	author='Aydin Buluc, John Gilbert, Adam Lugowski, Steve Reinhardt',
	url='http://kdt.sourceforge.net',
#	packages=['kdt', 'kdt'],
	ext_modules=[pyCombBLAS_ext],
	py_modules = ['kdt.pyCombBLAS', 'kdt.Graph', 'kdt.DiGraph', 'kdt.feedback', 'kdt.UFget'],
    extra_compile_args = ['/Od /Z7'],
    extra_link_args = ['/DEBUG']
    #	script_args=copy_args
	)
	
