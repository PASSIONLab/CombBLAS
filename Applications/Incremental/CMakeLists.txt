# Top level directory has the include files

#ADD_EXECUTABLE(inc Incremental.cpp)
ADD_EXECUTABLE(lcc LargestCC.cpp)
ADD_EXECUTABLE(inc-base Incremental-Baseline.cpp)
ADD_EXECUTABLE(inc-opt Incremental-Optimized.cpp)
ADD_EXECUTABLE(inc-full Incremental-Full.cpp)

#TARGET_LINK_LIBRARIES( inc CombBLAS)
TARGET_LINK_LIBRARIES( lcc CombBLAS)
TARGET_LINK_LIBRARIES( inc-base CombBLAS)
TARGET_LINK_LIBRARIES( inc-opt CombBLAS)
TARGET_LINK_LIBRARIES( inc-full CombBLAS)
