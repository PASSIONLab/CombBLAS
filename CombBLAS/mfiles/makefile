CFLAGS = -fopenmp -Drestrict=__restrict__ -O2 #-DNDEBUG 
COMBBLAS = ..

mmio.o: mmio.c
	gcc-5 $(CFLAGS) -Wno-write-strings -c -o mmio.o mmio.c

TOMMYS = $(COMBBLAS)/Tommy/tommyhashdyn.o $(COMBBLAS)/Tommy/tommyhash.o $(COMBBLAS)/Tommy/tommylist.o

$(COMBBLAS)/Tommy/%.o : $(COMBBLAS)/Tommy/%.c
	$(CXX) $(CXXFLAGS) -o $@ -c $<

mcl2mtx: mcl2mtx.cpp ThreadedMMConverter.h mmio.o $(TOMMYS)
	gcc-5 -std=c++11 $(CFLAGS) -o mcl2mtx mmio.o $(TOMMYS) mcl2mtx.cpp -lm -lstdc++

clean:
	rm mcl2mtx
	rm *.o
