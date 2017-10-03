# Top level directory has the include files


ADD_EXECUTABLE( SpMSpVBench SpMSpVBench.cpp )

TARGET_LINK_LIBRARIES( SpMSpVBench CommGridlib MPITypelib MemoryPoollib GraphGenlib Hashlib mmiolib)
