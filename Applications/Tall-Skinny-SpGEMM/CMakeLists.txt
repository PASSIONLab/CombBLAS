# Top level directory has the include files

## Add timing flag for timing breakdown prints
## https://stackoverflow.com/questions/44284275/passing-compiler-options-in-cmake
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -DTIMING")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DTIMING")

ADD_EXECUTABLE(summa_test summa_test.cpp)
ADD_EXECUTABLE(msbfs msbfs.cpp)

TARGET_LINK_LIBRARIES( summa_test CombBLAS )
TARGET_LINK_LIBRARIES( msbfs CombBLAS )
