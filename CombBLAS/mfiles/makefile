CFLAGS = -Drestrict=__restrict__ -O2 -fopenmp -DNDEBUG 
COMBBLAS = ..
CC = g++-5

gzstream/libgzstream.a: 
	$(MAKE) -C gzstream

mmio.o: mmio.c
	$(CC) $(CFLAGS) -Wno-write-strings -c -o mmio.o mmio.c

TOMMYS = $(COMBBLAS)/Tommy/tommyhashdyn.o $(COMBBLAS)/Tommy/tommyhash.o $(COMBBLAS)/Tommy/tommylist.o

$(COMBBLAS)/Tommy/%.o : $(COMBBLAS)/Tommy/%.c
	$(CC) $(CXXFLAGS) -o $@ -c $<

mcl2mtx: mcl2mtx.cpp MMConverter.h mmio.o $(TOMMYS) gzstream/libgzstream.a
	$(CC) -std=c++11 $(CFLAGS) -o mcl2mtx mmio.o $(TOMMYS) mcl2mtx.cpp -Lgzstream -lgzstream -lz

cc: cc.cpp MMConverter.h mmio.o $(TOMMYS) gzstream/libgzstream.a
	$(CC) -std=c++11 $(CFLAGS) -o cc mmio.o $(TOMMYS) cc.cpp -Lgzstream -lgzstream -lz

clean:
	rm -f mcl2mtx
	rm -f *.o
	rm -f ../Tommy/*.o
