#!/usr/bin/env python

#export CC=mpicxx
#export CXX=mpicxx

from distutils.core import setup, Extension
from distutils import sysconfig
import sys

print "Remember to set your preferred MPI C++ compiler in the CC and CXX environment variables. For example,"
print "export CC=mpicxx"
print "export CXX=mpicxx"
print ""

def check_for_header(header):
	from distutils.ccompiler import new_compiler, CompileError
	from shutil import rmtree
	import tempfile
	import os
	
	tmpdir = tempfile.mkdtemp()
	old = os.getcwd()
	
	os.chdir(tmpdir)
	
	# Try to include the relevant iXxx.h header, and disable the module
	# if it can't be found
	f = open('headertest.cpp', 'w')
	f.write("#include <%s>\n" % header)
	f.close()
	try:
		sys.stdout.write("Checking for %s... " % header)
		#include_dirs = self.include_dirs + ext.include_dirs
		new_compiler().compile([f.name])#, include_dirs=include_dirs)
		success = True
		sys.stdout.write("OK\n");
	except CompileError:
		sys.stdout.write("Not found\n");
		success = False
	
	os.chdir(old)
	rmtree(tmpdir)
	return success

# see if the compiler has TR1
hasTR1 = False
hasBoost = False
print "Checking for TR1..."
if (check_for_header("tr1/memory") and check_for_header("tr1/tuple")):
	hasTR1 = True
else:
	# nope, see if boost is available
	print "No TR1. Checking for Boost instead..."
	if (check_for_header("boost/tr1/memory") and check_for_header("boost/tr1/tuple")):
		hasBoost = True
	else:
		# nope, then sorry
		print "KDT uses features from TR1. These are available from some compilers or through the Boost C++ library (www.boost.org). Please make sure Boost is in your include path."
		sys.exit();

COMBBLAS = "CombBLAS/"
PCB = "kdt/pyCombBLAS/"
GENERATOR = "CombBLAS/graph500-1.2/generator/"

generator_files = [GENERATOR+"btrd_binomial_distribution.c", GENERATOR+"splittable_mrg.c", GENERATOR+"mrg_transitions.c", GENERATOR+"graph_generator.c", GENERATOR+"permutation_gen.c", GENERATOR+"make_graph.c", GENERATOR+"utils.c", GENERATOR+"scramble_edges.c"]

pyCombBLAS_ext = Extension('kdt._pyCombBLAS',
	[PCB+"pyCombBLAS.cpp", PCB+"pyCombBLAS_wrap.cpp", PCB+"pyDenseParVec.cpp", PCB+"pySpParVec.cpp", PCB+"pySpParMat.cpp", PCB+"pyOperations.cpp", COMBBLAS+"CommGrid.cpp", COMBBLAS+"MPIType.cpp", COMBBLAS+"MemoryPool.cpp"] + generator_files,
	#, include_dirs=['/usr/include/X11'],
	define_macros=[('NDEBUG', '1'),('restrict', '__restrict__'),('GRAPH_GENERATOR_SEQ', '1')],
	depends=[PCB+"pyCombBLAS.cpp"])

setup(name='kdt',
	version='0.1',
	description='Knowledge Discovery Toolbox',
	author='Aydin Buluc, John Gilbert, Adam Lugowski, Steve Reinhardt',
	url='http://kdt.sourceforge.net',
#	packages=['kdt', 'kdt'],
	ext_modules=[pyCombBLAS_ext],
	py_modules = ['kdt.pyCombBLAS', 'kdt.Graph', 'kdt.DiGraph', 'kdt.Graph500', 'kdt.feedback']
	#package_dir={"pyCombBLAS": 'Python/pyCombBLAS/'}
	)
	
