# module load cray-trilinos
CXX := CC
CXXFLAGS := -std=c++11 -O2 -DNDEBUG -fopenmp #-DDEBUG 

all: trilinosMatSquare 

clean:
	rm -rf trilinosMatSquare
	rm -f *.o



trilinosMatSquare: trilinosMatSquare.cpp 
	$(CXX) $(CXXFLAGS) trilinosMatSquare.cpp -DEPETRA_MPI -o trilinosMatSquare
