CFLAGS = -fopenmp -Drestrict=__restrict__ -O2 #-DNDEBUG 

mmio.o: mmio.c
	gcc-5 $(CFLAGS) -Wno-write-strings -c -o mmio.o mmio.c

mcl2mtx: mcl2mtx.cpp ThreadedMMConverter.h mmio.o
	gcc-5 -std=c++11 $(CFLAGS) -o mcl2mtx mmio.o mcl2mtx.cpp -lm -lstdc++

clean:
	rm mcl2mtx
	rm *.o