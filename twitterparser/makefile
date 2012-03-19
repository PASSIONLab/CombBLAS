$INCLUDE = /home/aydin/twitter/sparsehash-1.12/include
export OMP_NUM_THREADS=7

all:	parser remapper

parser:	monthlyparser.cpp
	g++ -std=c++0x -O3 -fopenmp -o parser monthlyparser.cpp

mapper:	remapper.cpp
	g++ -std=c++0x -O3 -fopenmp -o mapper remapper.cpp

balancer: balancer.cpp
	g++ -std=c++0x -O3 -fopenmp -o balancer balancer.cpp
