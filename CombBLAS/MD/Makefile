# C++ compiler
CXX := mpicxx
CXXFLAGS := -std=c++11 -O2 -fopenmp -fpermissive #-DDEBUG

all: md

COMBBLAS = ..

$(COMBBLAS)/graph500-1.2/generator/libgraph_generator_seq.a:
	$(MAKE) -C $(COMBBLAS)/graph500-1.2/generator

%.o : %.cpp 
	$(CXX) $(CXXFLAGS) -o $@ -c $<


clean:
	rm -rf ../*.o
	rm -f md
	rm -f *.o
	


md.o: MD.cpp ../SpDCCols.cpp ../dcsc.cpp ../SpHelper.h ../SpParMat.h ../ParFriends.h ../SpParMat.cpp ../SpDefs.h ../SpTuples.cpp ../DenseParVec.h ../DenseParVec.cpp ../SpImpl.h
	$(CXX) $(CXXFLAGS) -o $@ -c $<

md: MD.o $(COMBBLAS)/MPIType.o $(COMBBLAS)/MemoryPool.o $(COMBBLAS)/CommGrid.o  $(COMBBLAS)/hash.o $(COMBBLAS)/graph500-1.2/generator/libgraph_generator_seq.a $(COMBBLAS)/CommGrid.o  
	$(CXX) $(CXXFLAGS) -o $@ $^ -L$(COMBBLAS)/graph500-1.2/generator -lgraph_generator_seq

