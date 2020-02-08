# Top level directory has the include files

SET(MCL_CUDA_FLAGS "-std=c++14 -ccbin ${CMAKE_CXX_COMPILER} -O2 \
-Wno-deprecated-declarations -w -DNVCC -DSHOW_MEMORY_USAGE -DTIMING \
-Xcompiler -std=c++11 -Xcompiler -std=c++14 -Xcompiler -fopenmp \
-Xcompiler -DNDEBUG -Xcompiler -w -Xcompiler -DGRAPH_GENERATOR_SEQ \
-Xcompiler -O2 -Xcompiler -DMPICH_IGNORE_CXX_SEEK -Xcompiler -DTHREADED \
-Xcompiler -DSHOW_MEMORY_USAGE -Xcompiler -DTIMING")
SET(CMAKE_CUDA_FLAGS "${MCL_CUDA_FLAGS}")

ADD_EXECUTABLE( tdbfs TopDownBFS.cpp )
ADD_EXECUTABLE( dobfs DirOptBFS.cpp )
ADD_EXECUTABLE( fbfs FilteredBFS.cpp )
ADD_EXECUTABLE( fmis FilteredMIS.cpp )
ADD_EXECUTABLE( mcl_gpu MCL.cu )
ADD_EXECUTABLE( betwcent BetwCent.cpp )

TARGET_INCLUDE_DIRECTORIES ( mcl_gpu PRIVATE ../gspgemm ../gspgemm/nsparse ../gspgemm/rmerge ../gspgemm/rmerge/SPMMProject )

TARGET_LINK_LIBRARIES( tdbfs CombBLAS)
TARGET_LINK_LIBRARIES( dobfs CombBLAS)
TARGET_LINK_LIBRARIES( fbfs CombBLAS)
TARGET_LINK_LIBRARIES( fmis CombBLAS)
TARGET_LINK_LIBRARIES( mcl_gpu CombBLAS rmerge ${CUDA_cusparse_LIBRARY} )
TARGET_LINK_LIBRARIES( betwcent CombBLAS)

ADD_TEST(NAME BetwCent_Test COMMAND ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 4 $<TARGET_FILE:betwcent> ../TESTDATA/SCALE16BTW-TRANSBOOL/ 10 96 )
ADD_TEST(NAME TopDownBFS_Test COMMAND ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 4 $<TARGET_FILE:tdbfs> Force 17 FastGen)
ADD_TEST(NAME DirOptBFS_Test COMMAND ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 4 $<TARGET_FILE:dobfs> 17 )
ADD_TEST(NAME FBFS_Test COMMAND ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 4 $<TARGET_FILE:fbfs> Gen 16 )
ADD_TEST(NAME FMIS_Test COMMAND ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 4 $<TARGET_FILE:fmis> 17 )

