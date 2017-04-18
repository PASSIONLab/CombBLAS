CC = g++ 
FLAGS = -fopenmp -O2 -DNDEBUG 

all: mclconvert fscore

mclconvert.o: MCLConvert.cpp
	$(CC) $(FLAGS) -c -o mclconvert.o MCLConvert.cpp 
fscore.o: Fscore.cpp
	$(CC) $(FLAGS) -c -o fscore.o Fscore.cpp 



mclconvert: mclconvert.o 
	$(CC) $(FLAGS) -o mclconvert mclconvert.o 
fscore: fscore.o 
	$(CC) $(FLAGS) -o fscore fscore.o



clean:
	-rm -f mclconvert fscore *.o
