INCDIR = $(HOME)
INCADD = -I$(INCDIR) -I/usr/include/python2.4
COMBBLAS = ../../trunk/CombBLAS

OPT = -DNDEBUG -O3 -DMPICH_IGNORE_CXX_SEEK -Wreturn-type -w 
DEB = -g -O0 -fno-inline -DMPICH_IGNORE_CXX_SEEK -DTR1 #-DDEBUG
COMPILER = mpicxx
FLAGS = $(OPT) -fPIC -DCOMBBLAS_TR1
LINKFLAGS = #-undefined dynamic_lookup

all: tdsp 

%.o : %.c %.h
	$(COMPILER) $(INCADD) $(FLAGS) -c $< -o $@

#
# build CombBLAS
#

CommGrid.o:	$(COMBBLAS)/CommGrid.cpp $(COMBBLAS)/CommGrid.h
	$(COMPILER) $(INCADD) $(FLAGS) -c -o CommGrid.o $(COMBBLAS)/CommGrid.cpp

MPIType.o:	$(COMBBLAS)/MPIType.cpp $(COMBBLAS)/MPIType.h
	$(COMPILER) $(INCADD) $(FLAGS) -c -o MPIType.o $(COMBBLAS)/MPIType.cpp

MemoryPool.o: $(COMBBLAS)/MemoryPool.cpp $(COMBBLAS)/SpDefs.h
	$(COMPILER) $(INCADD) $(FLAGS) -c -o MemoryPool.o $(COMBBLAS)/MemoryPool.cpp


# build tdsp

tdsp.o: tdsp.cpp $(COMBBLAS)/CombBLAS.h Node.h Ft.h
	$(COMPILER) $(INCADD) $(FLAGS) -c -o tdsp.o tdsp.cpp

tdsp:	MemoryPool.o CommGrid.o MPIType.o tdsp.o
	$(COMPILER) $(INCADD) $(FLAGS) -o tdsp tdsp.o MemoryPool.o CommGrid.o MPIType.o


clean:
	rm -f tdsp 
	rm -f *.o
	rm -f *.a
	rm -f temp* 

cleanout:
	rm out.*
	rm err.*
