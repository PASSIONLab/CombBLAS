INCDIR = 
INCADD = -I$(INCDIR) 

OPT = -DNDEBUG -O3 -fno-exceptions 
DEB = -g -O0 -fno-inline
COMPILER = g++ #mpic++	

MemoryPool.o:	MemoryPool.cpp SpDefs.h
	$(COMPILER) $(DEB) -c -o MemoryPool.o MemoryPool.cpp

Driver.o: Driver.cpp SpDCCols.cpp dcsc.cpp SpHelper.h 
	$(COMPILER) $(DEB) -c -o Driver.o Driver.cpp

driver: Driver.o MemoryPool.o
	$(COMPILER) -g -o driver Driver.o MemoryPool.o

clean:
	rm -f driver
	rm -f Driver.o
	rm -f MemoryPool.o 

cleanout:
	rm out.*
	rm err.*
