INCDIR = /opt/local/include
INCADD = -I$(INCDIR) -I$(INCDIR)/mpich2

OPT = -DNDEBUG -O3 -DMPICH_IGNORE_CXX_SEEK
DEB = -g -O0 -fno-inline -DMPICH_IGNORE_CXX_SEEK
COMPILER = mpicxx	
FLAGS = $(DEB)

CommGrid.o:	../CommGrid.cpp ../CommGrid.h
	$(COMPILER) $(FLAGS) $(INCADD) -c -o CommGrid.o ../CommGrid.cpp

MPIType.o:	../MPIType.cpp ../MPIType.h
	$(COMPILER) $(FLAGS) $(INCADD) -c -o MPIType.o ../MPIType.cpp

MemoryPool.o:	../MemoryPool.cpp ../SpDefs.h
	$(COMPILER) $(FLAGS) $(INCADD) -c -o MemoryPool.o ../MemoryPool.cpp

TransposeTest.o: TransposeTest.cpp ../SpDCCols.cpp ../dcsc.cpp ../SpHelper.h ../SpParHelper.h ../SpParMat.cpp ../Friends.h ../ParFriends.h 
	$(COMPILER) $(INCADD) $(FLAGS) -c -o TransposeTest.o TransposeTest.cpp

MultTest.o: MultTest.cpp ../SpDCCols.cpp ../dcsc.cpp ../SpHelper.h ../SpParHelper.h ../SpParMat.cpp ../Friends.h ../ParFriends.h 
	$(COMPILER) $(INCADD) $(FLAGS) -c -o MultTest.o MultTest.cpp

TransposeTest: MemoryPool.o CommGrid.o MPIType.o TransposeTest.o
	$(COMPILER) $(FLAGS) $(INCADD) -o TransposeTest TransposeTest.o MemoryPool.o CommGrid.o MPIType.o

MultTest: MemoryPool.o CommGrid.o MPIType.o MultTest.o
	$(COMPILER) $(FLAGS) $(INCADD) -o MultTest MultTest.o MemoryPool.o CommGrid.o MPIType.o

clean:
	rm -f TransposeTest
	rm -f MultTest
	rm -f *.o

cleanout:
	rm out.*
	rm err.*
