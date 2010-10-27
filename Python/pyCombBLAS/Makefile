INCDIR = $(HOME)
INCADD = -I$(INCDIR) -I/usr/include/python2.4
COMBBLAS = ../../CombBLAS

# notes for configure:
# -fno-exceptions does not work with MPICH2
# -fno-rtti does not work with tr1:tuples

OPT = -DNDEBUG -O3 -DMPICH_IGNORE_CXX_SEEK
DEB = -g -O0 -fno-inline -DMPICH_IGNORE_CXX_SEEK #-DDEBUG
COMPILER = mpicxx
FLAGS = $(DEB) -fPIC 

all: DiGraph.py

DiGraph_wrap.cpp: DiGraph.i
	swig -python -c++ -shadow -module DiGraph -o DiGraph_wrap.cpp DiGraph.i

DiGraph_wrap.o: DiGraph_wrap.cpp
	$(COMPILER) $(INCADD) $(FLAGS) -c $< -o $@

%.o : %.cpp %.h
	$(COMPILER) $(INCADD) $(FLAGS) -c $< -o $@


# The rules below create a CombBLAS statically linked library for the non-templated functions.
# Maybe we should move this to CombBLAS?

CommGrid.o:	$(COMBBLAS)/CommGrid.cpp $(COMBBLAS)/CommGrid.h
	$(COMPILER) $(INCADD) $(FLAGS) -c -o CommGrid.o $(COMBBLAS)/CommGrid.cpp

MPIType.o:	$(COMBBLAS)/MPIType.cpp $(COMBBLAS)/MPIType.h
	$(COMPILER) $(INCADD) $(FLAGS) -c -o MPIType.o $(COMBBLAS)/MPIType.cpp

MemoryPool.o:	$(COMBBLAS)/MemoryPool.cpp $(COMBBLAS)/SpDefs.h
	$(COMPILER) $(INCADD) $(FLAGS) -c -o MemoryPool.o $(COMBBLAS)/MemoryPool.cpp	

libCombBLAS.a: CommGrid.o MPIType.o MemoryPool.o
	ar rcs libCombBLAS.a CommGrid.o MPIType.o MemoryPool.o



_DiGraph.so: DiGraph.o DiGraph_wrap.o libCombBLAS.a
	$(COMPILER) $(INCADD) $(FLAGS) -shared -o _DiGraph.so DiGraph.o DiGraph_wrap.o -L. -lCombBLAS

# the swig command that creates DiGraph_wrap.cpp also creates DiGraph.py
DiGraph.py:	_DiGraph.so



clean:
	rm -f *.so
	rm -f *.o
	rm -f *.a
	rm -f *.pyc
	rm -f DiGraph_wrap.cpp
	rm -f DiGraph.py

cleanout:
	rm out.*
	rm err.*
