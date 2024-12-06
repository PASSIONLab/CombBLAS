# Top level directory has the include files

## Add timing flag for timing breakdown prints
## https://stackoverflow.com/questions/44284275/passing-compiler-options-in-cmake
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -DTIMING")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DTIMING")

ADD_EXECUTABLE(inc Main.cpp)
ADD_EXECUTABLE(full Full.cpp)
ADD_EXECUTABLE(testideas Test.cpp)
ADD_EXECUTABLE(prep-data Prep-Data.cpp)
ADD_EXECUTABLE(prep-data-metaclust Prep-Data-Metaclust.cpp)
#ADD_EXECUTABLE(lcc LargestCC.cpp)
ADD_EXECUTABLE(inc-pipeline Incremental-Pipeline.cpp)
ADD_EXECUTABLE(inc-baseline-pipeline Incremental-Baseline-Pipeline.cpp)
ADD_EXECUTABLE(inc-toy-pipeline Incremental-Toy-Pipeline.cpp)

TARGET_LINK_LIBRARIES( inc CombBLAS )
TARGET_LINK_LIBRARIES( full CombBLAS )
TARGET_LINK_LIBRARIES( testideas CombBLAS )
TARGET_LINK_LIBRARIES( prep-data CombBLAS )
TARGET_LINK_LIBRARIES( prep-data-metaclust CombBLAS )
#TARGET_LINK_LIBRARIES( lcc CombBLAS )
TARGET_LINK_LIBRARIES( inc-pipeline CombBLAS )
TARGET_LINK_LIBRARIES( inc-baseline-pipeline CombBLAS )
TARGET_LINK_LIBRARIES( inc-toy-pipeline CombBLAS )
