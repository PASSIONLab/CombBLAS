INCLUDE(CheckIncludeFileCXX)

cmake_minimum_required(VERSION 2.0)
PROJECT( COMBINATORIAL_BLAS )
SET(CMAKE_CXX_COMPILER CC) 
SET(CMAKE_INCLUDE_CURRENT_DIR ON)
SET(CMAKE_VERBOSE_MAKEFILE ON)

CHECK_INCLUDE_FILE_CXX(tr1/tuple HAVE_TR1)

if (NOT HAVE_TR1)
	FIND_PACKAGE(Boost)
	MESSAGE(${Boost_INCLUDE_DIR})
	#FIND_PATH(Boost_INCLUDE_DIR
	#		NAMES boost/config.hpp
	#		PATH_SUFFIXES boost-1_33 boost-1_34 boost-1_35
	#		boost-1_36 boost-1_37 boost-1_38 boost-1_39 boost-1_40)	
	SET(CMAKE_CXX_FLAGS "-I${Boost_INCLUDE_DIR} -DNDEBUG -O3 -DNOTR1 -DMPICH_IGNORE_CXX_SEEK")

	IF (NOT Boost_INCLUDE_DIR)
		SET(BOOST_ROOT $ENV{TACC_BOOST_INC})
		MESSAGE("Searching under ${BOOST_ROOT}")
		FIND_PACKAGE(Boost)
        	MESSAGE(${Boost_INCLUDE_DIR})
	ENDIF (NOT Boost_INCLUDE_DIR)

	SET(CMAKE_CXX_FLAGS "-I${Boost_INCLUDE_DIR} -DNDEBUG -O3 -DNOTR1 -DMPICH_IGNORE_CXX_SEEK")

	IF (NOT Boost_INCLUDE_DIR)
        	MESSAGE(FATAL_ERROR "Cannot find boost or compliant libstdc++")
	ENDIF (NOT Boost_INCLUDE_DIR)

else (NOT HAVE_TR1)
	SET(CMAKE_CXX_FLAGS "-DNDEBUG -DGRAPH_GENERATOR_SEQ -O3 -DMPICH_IGNORE_CXX_SEEK")
	SET(CMAKE_C_FLAGS "-DNDEBUG -DGRAPH_GENERATOR_SEQ -O3 -DMPICH_IGNORE_CXX_SEEK")
endif (NOT HAVE_TR1)

ADD_LIBRARY( CommGridlib CommGrid.cpp  )
ADD_LIBRARY( MPITypelib MPIType.cpp )
ADD_LIBRARY( MemoryPoollib MemoryPool.cpp )

ADD_SUBDIRECTORY( ReleaseTests )
ADD_SUBDIRECTORY( Applications ) 
ADD_SUBDIRECTORY( graph500-1.2/generator )

ENABLE_TESTING()
INCLUDE(CTest)
ADD_TEST( Multiplication_Test aprun -n 4 ReleaseTests/MultTest TESTDATA/rmat_scale16_A.txt TESTDATA/rmat_scale16_B.txt TESTDATA/rmat_scale16_productAB.txt TESTDATA/x_65536_halfdense.txt TESTDATA/y_65536_halfdense.txt)
ADD_TEST( Reduction_Test aprun -n 4 ReleaseTests/ReduceTest TESTDATA/sprand10000 TESTDATA/sprand10000_sumcols TESTDATA/sprand10000_sumrows)
ADD_TEST( Iterator_Test aprun -n 4 ReleaseTests/IteratorTest TESTDATA sprand10000)
ADD_TEST( Transpose_Test aprun -n 4 ReleaseTests/TransposeTest TESTDATA betwinput_scale16 betwinput_transposed_scale16)
ADD_TEST( Indexing_Test aprun -n 4 ReleaseTests/IndexingTest TESTDATA B_100x100.txt B_10x30_Indexed.txt rand10outta100.txt rand30outta100.txt)
ADD_TEST( FindSparse_Test aprun -n 4 ReleaseTests/FindSparse TESTDATA findmatrix.txt)
ADD_TEST( Scale_Timing aprun -n 4 ReleaseTests/ScaleTiming TESTDATA/matrix100.txt TESTDATA/scaler.txt TESTDATA/matrix100_scaled.txt)
