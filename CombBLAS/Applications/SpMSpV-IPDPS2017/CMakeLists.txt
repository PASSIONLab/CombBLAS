# Top level directory has the include files


ADD_EXECUTABLE( SpMSpVBench SpMSpVBench.cpp )

TARGET_LINK_LIBRARIES( SpMSpVBench CombBLAS)

ADD_TEST(NAME SpMSpVBench_test COMMAND ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 4 $<TARGET_FILE:SpMSpVBench> -rmat 18 )

