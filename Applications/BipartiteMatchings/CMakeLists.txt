# Top level directory has the include files


ADD_EXECUTABLE( bpml BPMaximalMatching.cpp )
ADD_EXECUTABLE( bpmm BPMaximumMatching.cpp )
ADD_EXECUTABLE( awpm ApproxWeightPerfectMatching.cpp )


TARGET_LINK_LIBRARIES( bpml CombBLAS)
TARGET_LINK_LIBRARIES( bpmm CombBLAS )
TARGET_LINK_LIBRARIES( awpm CombBLAS )

ADD_TEST(NAME BPML_Test COMMAND ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 4 $<TARGET_FILE:bpml> er 12 8  )
ADD_TEST(NAME BPMM_Test COMMAND ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 4 $<TARGET_FILE:bpmm> er 12 8  )
