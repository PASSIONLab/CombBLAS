INCLUDE(CheckIncludeFileCXX)

cmake_minimum_required(VERSION 2.0)
PROJECT( COMBINATORIAL_BLAS )
SET(CMAKE_CXX_COMPILER mpicxx) 
SET(CMAKE_INCLUDE_CURRENT_DIR ON)
SET(CMAKE_VERBOSE_MAKEFILE ON)

CHECK_INCLUDE_FILE_CXX(tr1/tuple HAVE_TR1)

if (NOT HAVE_TR1)
	FIND_PACKAGE(Boost REQUIRED)
	FIND_PATH(BOOST_INCLUDE_DIR
			NAMES boost/config.hpp
			PATH_SUFFIXES boost-1_33 boost-1_39 boost-1_34 boost-1_35 boost-1_36 boost-1_37 boost-1_38)	
	SET(CMAKE_CXX_FLAGS "-I{BOOST_INCLUDE_DIR} -DNDEBUG -O3 -DNOTR1 -DMPICH_IGNORE_CXX_SEEK")
else (NOT HAVE_TR1)
	SET(CMAKE_CXX_FLAGS "-DNDEBUG -O3 -DMPICH_IGNORE_CXX_SEEK")
endif (NOT HAVE_TR1)

#ADD_SUBDIRECTORY( Applications ) 
ADD_SUBDIRECTORY( ReleaseTests )

ENABLE_TESTING()
INCLUDE(CTest)
ADD_TEST( Multiplication_Test mpirun -np 4 ReleaseTests/MultTest TESTDATA/rmat_scale16_A.txt TESTDATA/rmat_scale16_B.txt TESTDATA/rmat_scale16_productAB.txt)
ADD_TEST( Reduction_Test mpirun -np 4 ReleaseTests/ReduceTest TESTDATA/sprand10000 TESTDATA/sprand10000_sumcols TESTDATA/sprand10000_sumrows)
ADD_TEST( Iterator_Test mpirun -np 4 ReleaseTests/IteratorTest TESTDATA sprand10000)

