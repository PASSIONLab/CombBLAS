CFLAGS = -Drestrict=__restrict__ -O2 -fopenmp -DNDEBUG 
COMBBLAS = ../include
CC = g++

gzstream/libgzstream.a: 
	$(MAKE) -C gzstream

mmio.o: mmio.c
	$(CC) $(CFLAGS) -Wno-write-strings -c -o mmio.o mmio.c

TOMMYS = $(COMBBLAS)/Tommy/tommyhashdyn.o $(COMBBLAS)/Tommy/tommyhash.o $(COMBBLAS)/Tommy/tommylist.o

$(COMBBLAS)/Tommy/%.o : $(COMBBLAS)/Tommy/%.c
	$(CC) $(CXXFLAGS) -o $@ -c $<

mcl2mtx_sub: mcl2mtx.cpp MMConverter.h mmio.o $(TOMMYS) gzstream/libgzstream.a
	$(CC) -std=c++11 $(CFLAGS) -DSUBGRAPHS -o mcl2mtx_sub mmio.o $(TOMMYS) mcl2mtx.cpp -Lgzstream -lgzstream -lz

mcl2mtx: mcl2mtx.cpp MMConverter.h mmio.o $(TOMMYS) gzstream/libgzstream.a
	$(CC) -std=c++11 $(CFLAGS) -I${COMBBLAS} -o mcl2mtx mmio.o $(TOMMYS) mcl2mtx.cpp -Lgzstream -lgzstream -lz

cc: cc.cpp MMConverter.h mmio.o $(TOMMYS) gzstream/libgzstream.a
	$(CC) -std=c++11 $(CFLAGS) -I${COMBBLAS} -o cc mmio.o $(TOMMYS) cc.cpp -Lgzstream -lgzstream -lz

mer2gr: mer_to_graph.cpp
	$(CC) -std=c++11 $(CFLAGS) -o mer2gr mer_to_graph.cpp 

clean:
	rm -f mcl2mtx
	rm -f *.o
	rm -f ../Tommy/*.o
	rm -f gzstream/libgzstream.a
