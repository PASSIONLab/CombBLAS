#!/usr/bin/env python

#export CC=mpicxx
#export CXX=mpicxx

from distutils.core import setup, Extension
from distutils import sysconfig

print "Remember to set your preferred MPI C++ compiler in the CC and CXX environment variables. For example,"
print "export CC=mpicxx"
print "export CXX=mpicxx"
print ""

# see if the compiler has TR1
#print "Checking for TR1..."
# nope, see if boost is available
#print "No TR1. Checking for Boost instead..."
# nope, then sorry
#print "KDT uses smart pointers which are available through the Boost library. Please make sure Boost is in your include path."

COMBBLAS = "CombBLAS/"
PCB = "Python/pyCombBLAS/"
GENERATOR = "CombBLAS/graph500-1.2/generator/"

generator_files = [GENERATOR+"btrd_binomial_distribution.c", GENERATOR+"splittable_mrg.c", GENERATOR+"mrg_transitions.c", GENERATOR+"graph_generator.c", GENERATOR+"permutation_gen.c", GENERATOR+"make_graph.c", GENERATOR+"utils.c", GENERATOR+"scramble_edges.c"]

pyCombBLAS_ext = Extension('_pyCombBLAS',
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
	py_modules = ['pyCombBLAS']
	#package_dir={"pyCombBLAS": 'Python/pyCombBLAS/'}
	)
	
