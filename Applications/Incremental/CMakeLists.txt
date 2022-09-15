# Top level directory has the include files

ADD_EXECUTABLE(inc Incremental.cpp)
ADD_EXECUTABLE(lcc LargestCC.cpp)
ADD_EXECUTABLE(prep-split PrepareTestSplits.cpp)

TARGET_LINK_LIBRARIES( inc CombBLAS)
TARGET_LINK_LIBRARIES( lcc CombBLAS)
TARGET_LINK_LIBRARIES( prep-split CombBLAS)
