CFLAGS = -Drestrict=__restrict__ -O2 -fopenmp #-DNDEBUG 
COMBBLAS = ..
CC = gcc-5

mmio.o: mmio.c
	$(CC) $(CFLAGS) -Wno-write-strings -c -o mmio.o mmio.c

TOMMYS = $(COMBBLAS)/Tommy/tommyhashdyn.o $(COMBBLAS)/Tommy/tommyhash.o $(COMBBLAS)/Tommy/tommylist.o

$(COMBBLAS)/Tommy/%.o : $(COMBBLAS)/Tommy/%.c
	$(CXX) $(CXXFLAGS) -o $@ -c $<

mcl2mtx: mcl2mtx.cpp MMConverter.h mmio.o $(TOMMYS)
	$(CC) -std=c++11 $(CFLAGS) -o mcl2mtx mmio.o $(TOMMYS) mcl2mtx.cpp -lm -lstdc++

clean:
	rm -f mcl2mtx
	rm -f *.o
	rm -f ../Tommy/*.o
