# Top level directory has the include files

ADD_EXECUTABLE( betwcent BetwCent.cpp )
ADD_EXECUTABLE( mcl MCL.cpp )

TARGET_LINK_LIBRARIES( mcl CommGridlib MPITypelib MemoryPoollib )
TARGET_LINK_LIBRARIES( betwcent CommGridlib MPITypelib MemoryPoollib )

