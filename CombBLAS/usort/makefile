
binUtils.o: src/binUtils.cpp include/binUtils.h
	mpicxx -I./include -std=c++11 -fopenmp -c src/binUtils.cpp

parUtils.o: src/parUtils.cpp include/parUtils.h
	mpicxx -I./include -std=c++11 -fopenmp -c src/parUtils.cpp

main:	parUtils.o binUtils.o src/main.cpp
	mpicxx -I./include -std=c++11 -fopenmp src/main.cpp binUtils.o parUtils.o -o main 

