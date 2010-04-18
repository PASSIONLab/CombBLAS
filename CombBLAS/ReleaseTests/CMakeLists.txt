# Top level directory has the include files

ADD_LIBRARY( CommGridlib ../CommGrid.cpp  )
ADD_LIBRARY( MPITypelib ../MPIType.cpp )
ADD_LIBRARY( MemoryPoollib ../MemoryPool.cpp )

ADD_EXECUTABLE( MultTiming MultTiming.cpp )
ADD_EXECUTABLE( MultTest MultTest.cpp )
ADD_EXECUTABLE( ReduceTest ReduceTest.cpp )
ADD_EXECUTABLE( TransposeTest TransposeTest.cpp )
ADD_EXECUTABLE( IteratorTest IteratorTest.cpp )

TARGET_LINK_LIBRARIES( MultTiming CommGridlib MPITypelib MemoryPoollib )
TARGET_LINK_LIBRARIES( MultTest CommGridlib MPITypelib MemoryPoollib )
TARGET_LINK_LIBRARIES( ReduceTest CommGridlib MPITypelib MemoryPoollib )
TARGET_LINK_LIBRARIES( TransposeTest CommGridlib MPITypelib MemoryPoollib )
TARGET_LINK_LIBRARIES( IteratorTest CommGridlib MPITypelib MemoryPoollib )

