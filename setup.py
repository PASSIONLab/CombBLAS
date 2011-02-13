#!/usr/bin/env python

#export CC=mpicxx
#export CXX=mpicxx

from distutils.core import setup, Extension
from distutils import sysconfig
import sys

print "Remember to set your preferred MPI C++ compiler in the CC and CXX environment variables. For example, in Bash:"
print "export CC=mpicxx"
print "export CXX=mpicxx"
print ""

def check_for_header(header, include_dirs):
	"""Check for the existence of a header file by creating a small program which includes it and see if it compiles."""
	from distutils.ccompiler import new_compiler, CompileError
	from shutil import rmtree
	import tempfile
	import os
	
	try:
		tmpdir = tempfile.mkdtemp()
	except AttributeError:
		# Python 2.2 doesn't have mkdtemp().
		tmpdir = "header_check_tempdir"
		try:
			os.mkdir(tmpdir)
		except OSError:
			print "Can't create temporary directory. Aborting."
			sys.exit()
			
	old = os.getcwd()
	
	os.chdir(tmpdir)
	
	# Try to include the header
	f = open('headertest.cpp', 'w')
	f.write("#include <%s>\n" % header)
	f.close()
	try:
		sys.stdout.write("Checking for <%s>... " % header)
		new_compiler().compile([f.name], include_dirs=include_dirs)
		success = True
		sys.stdout.write("OK\n");
	except CompileError:
		sys.stdout.write("Not found\n");
		success = False
	
	os.chdir(old)
	rmtree(tmpdir)
	return success
	
# parse out additional include dirs from the command line
include_dirs = []
copy_args=sys.argv[1:]
for a in copy_args:
	if a.startswith('-I'):
		include_dirs.append(a[2:])
		copy_args.remove(a)

# see if the compiler has TR1
hasTR1 = False
hasBoost = False
headerDefs = []
print "Checking for TR1..."
if (check_for_header("tr1/memory", include_dirs) and check_for_header("tr1/tuple", include_dirs)):
	hasTR1 = True
else:
	# nope, see if boost is available
	print "No TR1. Checking for Boost instead..."
	if (check_for_header("boost/tr1/memory.hpp", include_dirs) and check_for_header("boost/tr1/tuple.hpp", include_dirs)):
		hasBoost = True
		headerDefs = [('NOTR1', '1')]
	else:
		# nope, then sorry
		print "KDT uses features from C++ TR1. These are available from some compilers or through the Boost C++ library (www.boost.org)."
		print "Please make sure Boost is in your system include path or append the include path with the -I switch."
		print "For example, if you have Boost installed in /home/username/include/boost:"
		print "$ python setup.py build -I/home/username/include"
		sys.exit();




COMBBLAS = "CombBLAS/"
PCB = "kdt/pyCombBLAS/"
GENERATOR = "CombBLAS/graph500-1.2/generator/"

#files for the graph500 graph generator.
generator_files = [GENERATOR+"btrd_binomial_distribution.c", GENERATOR+"splittable_mrg.c", GENERATOR+"mrg_transitions.c", GENERATOR+"graph_generator.c", GENERATOR+"permutation_gen.c", GENERATOR+"make_graph.c", GENERATOR+"utils.c", GENERATOR+"scramble_edges.c"]

#pyCombBLAS extension which wraps the templated C++ Combinatorial BLAS library. 
pyCombBLAS_ext = Extension('kdt._pyCombBLAS',
	[PCB+"pyCombBLAS.cpp", PCB+"pyCombBLAS_wrap.cpp", PCB+"pyDenseParVec.cpp", PCB+"pySpParVec.cpp", PCB+"pySpParMat.cpp", PCB+"pyOperations.cpp", COMBBLAS+"CommGrid.cpp", COMBBLAS+"MPIType.cpp", COMBBLAS+"MemoryPool.cpp"] + generator_files,
	include_dirs=include_dirs,
	define_macros=[('NDEBUG', '1'),('restrict', '__restrict__'),('GRAPH_GENERATOR_SEQ', '1')] + headerDefs)

setup(name='kdt',
	version='0.1',
	description='Knowledge Discovery Toolbox',
	author='Aydin Buluc, John Gilbert, Adam Lugowski, Steve Reinhardt',
	url='http://kdt.sourceforge.net',
#	packages=['kdt', 'kdt'],
	ext_modules=[pyCombBLAS_ext],
	py_modules = ['kdt.pyCombBLAS', 'kdt.Graph', 'kdt.DiGraph', 'kdt.Graph500', 'kdt.feedback'],
	script_args=copy_args
	)
	
