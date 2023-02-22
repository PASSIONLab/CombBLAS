# Top level directory has the include files

ADD_EXECUTABLE(inc Main.cpp)
ADD_EXECUTABLE(testideas Test.cpp)
ADD_EXECUTABLE(prep-data Prep-Data.cpp)
ADD_EXECUTABLE(lcc LargestCC.cpp)
ADD_EXECUTABLE(inc-full Incremental-Full.cpp)
ADD_EXECUTABLE(inc-base Incremental-Baseline.cpp)
ADD_EXECUTABLE(inc-v1 Incremental-V1.cpp)
ADD_EXECUTABLE(inc-v2 Incremental-V2.cpp)

TARGET_LINK_LIBRARIES( inc CombBLAS )
TARGET_LINK_LIBRARIES( testideas CombBLAS )
TARGET_LINK_LIBRARIES( prep-data CombBLAS )
TARGET_LINK_LIBRARIES( lcc CombBLAS )
TARGET_LINK_LIBRARIES( inc-full CombBLAS )
TARGET_LINK_LIBRARIES( inc-base CombBLAS )
TARGET_LINK_LIBRARIES( inc-v1 CombBLAS )
TARGET_LINK_LIBRARIES( inc-v2 CombBLAS )
