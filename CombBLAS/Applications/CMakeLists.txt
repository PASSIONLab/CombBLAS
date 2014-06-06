# Top level directory has the include files

ADD_EXECUTABLE( betwcent BetwCent.cpp )
ADD_EXECUTABLE( mcl MCL.cpp )
ADD_EXECUTABLE( tdbfs TopDownBFS.cpp )
ADD_EXECUTABLE( dobfs DirOptBFS.cpp )
ADD_EXECUTABLE( fbfs FilteredBFS.cpp )
ADD_EXECUTABLE( fmis FilteredMIS.cpp )


TARGET_LINK_LIBRARIES( mcl CommGridlib MPITypelib MemoryPoollib Hashlib)
TARGET_LINK_LIBRARIES( betwcent CommGridlib MPITypelib MemoryPoollib Hashlib)
TARGET_LINK_LIBRARIES( tdbfs CommGridlib MPITypelib MemoryPoollib GraphGenlib Hashlib)
TARGET_LINK_LIBRARIES( dobfs CommGridlib MPITypelib MemoryPoollib GraphGenlib Hashlib)
TARGET_LINK_LIBRARIES( fbfs CommGridlib MPITypelib MemoryPoollib GraphGenlib Hashlib)
TARGET_LINK_LIBRARIES( fmis CommGridlib MPITypelib MemoryPoollib GraphGenlib Hashlib)


