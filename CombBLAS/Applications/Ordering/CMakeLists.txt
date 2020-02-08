# Top level directory has the include files

# MESSAGE("aaa " ${CMAKE_CXX_FLAGS})

ADD_EXECUTABLE( rcm RCM.cpp )

TARGET_LINK_LIBRARIES( rcm CombBLAS)

ADD_TEST(NAME RCM_Test COMMAND ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 4 $<TARGET_FILE:rcm> er 12 )

