#!/usr/bin/env python

#export CC=mpicxx
#export CXX=mpicxx

from distutils.core import setup, Extension
from distutils import sysconfig
import sys

COMBBLAS = "CombBLAS/"
PCB = "kdt/pyCombBLAS/"
GENERATOR = "CombBLAS/graph500-1.2/generator/"
debug = False

print "Remember to set your preferred MPI C++ compiler in the CC and CXX environment variables. For example, in Bash:"
print "export CC=mpicxx"
print "export CXX=mpicxx"
print ""

############################################################################
#### HELPER FUNCTIONS
def see_if_compiles(program, include_dirs, define_macros):
	""" Try to compile the passed in program and report if it compiles successfully or not. """
	from distutils.ccompiler import new_compiler, CompileError
	from shutil import rmtree
	import tempfile
	import os
	
	try:
		tmpdir = tempfile.mkdtemp()
	except AttributeError:
		# Python 2.2 doesn't have mkdtemp().
		tmpdir = "compile_check_tempdir"
		try:
			os.mkdir(tmpdir)
		except OSError:
			print "Can't create temporary directory. Aborting."
			sys.exit()
			
	old = os.getcwd()
	
	os.chdir(tmpdir)
	
	# Try to include the header
	f = open('compiletest.cpp', 'w')
	f.write(program)
	f.close()
	try:
		c = new_compiler()
		for macro in define_macros:
			c.define_macro(name=macro[0], value=macro[1])
		c.compile([f.name], include_dirs=include_dirs)
		success = True
	except CompileError:
		success = False
	
	os.chdir(old)
	rmtree(tmpdir)
	return success

def check_for_header(header, include_dirs, define_macros):
	"""Check for the existence of a header file by creating a small program which includes it and see if it compiles."""
	program = "#include <%s>\n" % header
	sys.stdout.write("Checking for <%s>... " % header)
	success = see_if_compiles(program, include_dirs, define_macros)
	if (success):
		sys.stdout.write("OK\n");
	else:
		sys.stdout.write("Not found\n");
	return success

############################################################################
#### INDIVIDUAL TEST FUNCTIONS

def check_for_MPI_IN_PLACE(include_dirs, define_macros):
	""" Check for the existence of the MPI_IN_PLACE constant. """
	
	program = """
#include <mpi.h>

int main(int argc, const char** argv) {
	void* buf = NULL;
	MPI_Allreduce(MPI_IN_PLACE, buf, 10, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
	return 0;
}

"""
	sys.stdout.write("Checking for MPI_IN_PLACE... ")
	sys.stdout.flush()
	success = see_if_compiles(program, include_dirs, define_macros)
	if (success):
		sys.stdout.write("OK\n");
	else:
		sys.stdout.write("Not found\n");
	return success

def check_for_C99_CONSTANTS(include_dirs, define_macros):
	""" See if C99 constants for integers are defined. """
	
	program = """
#include <iostream>
#include <stdint.h>

int main()
{
        uint64_t v, val0;
        v *= (val0 | UINT64_C(0x4519840211493211));
        uint32_t l = (uint32_t)(x & UINT32_MAX);
        return v;
}

"""
	sys.stdout.write("Checking for C99 constants... ")
	success = see_if_compiles(program, include_dirs, define_macros)
	if (success):
		sys.stdout.write("OK\n");
	else:
		sys.stdout.write("Not found, will use __STDC_CONSTANT_MACROS and __STDC_LIMIT_MACROS\n");
	return success


def check_for_Windows(include_dirs, define_macros):
	""" See if we are on Windows """
	
	program = """
#include <windows.h>

"""
	sys.stdout.write("Checking for Windows... ")
	success = see_if_compiles(program, include_dirs, define_macros)
	if (success):
		sys.stdout.write("Yes\n");
	else:
		sys.stdout.write("No\n");
	return success
	
def check_for_VS(include_dirs, define_macros):
	""" See if we are compiling with Visual C++ """
	
	program = """
static int foo = _MSC_VER;
"""
	sys.stdout.write("Checking for Visual C++... ")
	success = see_if_compiles(program, include_dirs, define_macros)
	if (success):
		sys.stdout.write("Yes\n");
	else:
		sys.stdout.write("No\n");
	return success

	
############################################################################
#### COMMAND LINE ARGS

# parse out additional include dirs from the command line
include_dirs = []
library_dirs = []
libraries = []
define_macros = [("MPICH_IGNORE_CXX_SEEK", None)]
extra_link_args = []
extra_compile_args = []
usingWinMPICH = False
usingWindows = False
MPICHdir = "C:\Program Files\MPICH2"
copy_args=sys.argv[1:]
for a in copy_args:
	if a.startswith('-I'):
		include_dirs.append(a[2:])
		copy_args.remove(a)
	if a.startswith('-MPICH'):
		usingWinMPICH = True
		usingWindows = True
		if a.startswith('-MPICH='):
			MPICHdir=a[7:]
		print "Using Windows MPICH from '%s'. On other platforms simply use MPICH's mpicxx compiler instead of specifying the -MPICH switch."%(MPICHdir)
		copy_args.remove(a)
	if a.startswith('-debug'):
		debug = True
		copy_args.remove(a)
	if a.startswith('-D'):
		# macros can be a single value or a constant=value pair
		macro = tuple(a[2:].split("="))
		if (len(macro) == 1):
			macro = (macro[0], None)
		define_macros.append(macro)
		copy_args.remove(a)

############################################################################
#### RUNNING TESTS

# see if the compiler has TR1
hasCpp0x = False
hasTR1 = False
hasBoost = False
headerDefs = []
print "Checking for C++0x:"
if check_for_header("memory", include_dirs, define_macros) and check_for_header("unordered_map", include_dirs, define_macros):
	hasCpp0x = True
else:
	print "No C++0x. Checking for TR1:"
	if check_for_header("tr1/memory", include_dirs, define_macros) and check_for_header("tr1/unordered_map", include_dirs, define_macros):
		hasTR1 = True
		headerDefs = [('COMBBLAS_TR1', None)]
	else:
		# nope, see if boost is available
		print "No TR1. Checking for Boost:"
		if check_for_header("boost/tr1/memory.hpp", include_dirs, define_macros) and check_for_header("boost/tr1/unordered_map.hpp", include_dirs, define_macros):
			hasBoost = True
			headerDefs = [('COMBBLAS_BOOST', None)]
		else:
			# nope, then sorry
			print "KDT uses features from C++0x (TR1). These are available from some compilers or through the Boost C++ library (www.boost.org)."
			print "Please make sure Boost is in your system include path or append the include path with the -I switch."
			print "For example, if you have Boost installed in /home/username/include/boost:"
			print "$ python setup.py build -I/home/username/include"
			sys.exit();

#if (not check_for_MPI_IN_PLACE(include_dirs, define_macros)):
#	print "Please use a more recent MPI implementation."
#	print "If your system has multiple MPI implementations you can set your preferred MPI C++ compiler in the CC and CXX environment variables. For example, in Bash:"
#	print "export CC=mpicxx"
#	print "export CXX=mpicxx"
#	sys.exit();

if not check_for_C99_CONSTANTS(include_dirs, define_macros):
	define_macros.append(("__STDC_CONSTANT_MACROS", None))
	define_macros.append(("__STDC_LIMIT_MACROS", None))

# Windows-specific things
if check_for_Windows(include_dirs, define_macros):
	usingWindows = True
	define_macros.append(("NOMINMAX", None))               # Windows defines min and max as macros, which wreaks havoc with functions named min and max, regardless of namespace
	if not usingWinMPICH:
		usingWinMPICH = True
		print "You are on Windows but have not specified MPICH with the -MPICH or the -MPICH=path switches. We only support KDT on Windows with MPICH, so we assume the default MPICH path of '%s'."%(MPICHdir)
	# add debug compiler flags?
	if debug:
		extra_compile_args.append('/Od')   # no optimizations, override the default /Ox
		extra_compile_args.append('/Zi')   # debugging info
		extra_link_args.append('/debug')   # debugging info
	
	if check_for_VS(include_dirs, define_macros):
		define_macros.append(('inline', '__inline'))
		define_macros.append(('_SCL_SECURE_NO_WARNINGS', '1'))    # disables odd but annoyingly verbose checks, maybe these are legit, don't know.
# still need for ('restrict', '__restrict__') define_macro on non-Windows?
		
if not check_for_header("inttypes.h", include_dirs, define_macros):
	include_dirs.append(COMBBLAS+"ms_inttypes")            # VS2008 does not include <inttypes.h>
if not check_for_header("sys/time.h", include_dirs, define_macros):
	include_dirs.append(COMBBLAS+"ms_sys")                 # VS2008 does not include <sys/time.h>, we include a blank one because other people's code includes it but none of the functions are used.

if usingWinMPICH:
	include_dirs.append(MPICHdir + "\include")
	library_dirs.append(MPICHdir + "\lib")
	libraries.append("mpi")
	libraries.append("cxx")


############################################################################
#### RUN DISTUTILS

#files for the graph500 graph generator.
generator_files = [GENERATOR+"btrd_binomial_distribution.c", GENERATOR+"splittable_mrg.c", GENERATOR+"mrg_transitions.c", GENERATOR+"graph_generator.c", GENERATOR+"permutation_gen.c", GENERATOR+"make_graph.c", GENERATOR+"utils.c", GENERATOR+"scramble_edges.c"]

#pyCombBLAS extension which wraps the templated C++ Combinatorial BLAS library. 
pyCombBLAS_ext = Extension('kdt._pyCombBLAS',
	[PCB+"pyCombBLAS.cpp", PCB+"pyCombBLAS_wrap.cpp", PCB+"pyDenseParVec.cpp", PCB+"pyObjDenseParVec.cpp", PCB+"pySpParVec.cpp", PCB+"pySpParMat.cpp", PCB+"pySpParMatBool.cpp", PCB+"pyOperations.cpp", COMBBLAS+"CommGrid.cpp", COMBBLAS+"MPIType.cpp", COMBBLAS+"MemoryPool.cpp"] + generator_files,
	include_dirs=include_dirs,
	library_dirs=library_dirs,
	libraries=libraries,
	extra_link_args = extra_link_args, extra_compile_args = extra_compile_args,
	define_macros=[('NDEBUG', '1'),('GRAPH_GENERATOR_SEQ', '1')] + headerDefs + define_macros)

setup(name='kdt',
	version='0.1',
	description='Knowledge Discovery Toolbox',
	author='Aydin Buluc, John Gilbert, Adam Lugowski, Steve Reinhardt',
	url='http://kdt.sourceforge.net',
#	packages=['kdt', 'kdt'],
	ext_modules=[pyCombBLAS_ext],
	py_modules = ['kdt.pyCombBLAS', 'kdt.Graph', 'kdt.DiGraph', 'kdt.HyGraph', 'kdt.feedback', 'kdt.UFget'],
	script_args=copy_args
	)
	
