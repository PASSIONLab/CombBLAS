# Top level directory has the include files


ADD_EXECUTABLE( rcm RCM.cpp )

TARGET_LINK_LIBRARIES( rcm CommGridlib MPITypelib MemoryPoollib GraphGenlib Hashlib mmiolib MPIOplib)
