ifeq ($(CXX),$(filter $(CXX),g++ gcc))
MYFAST_FLAGS = -O3
else
MYFAST_FLAGS = -O3 -ip -fp-model fast=2
endif

NVCC_FLAGS := -Wno-deprecated-declarations -O3

all: lib
cuSpmmTest.o:
	$(CUDA_CC) $(NVCC_FLAGS) -c cuSpmmTest.cu -o cuSpmmTest.o -I .. -I $(BOOST_INC) -m64 -std=c++14 $(CUDA_INC)
cuSpmmOther.o:
	$(CUDA_CC) $(NVCC_FLAGS) -c cuSpmmOther.cu -o cuSpmmOther.o -I .. -I $(BOOST_INC) -m64 -std=c++14 $(CUDA_INC)
bhsparse.o:
	$(CUDA_CC) $(NVCC_FLAGS) -c bhsparse.cu -o bhsparse.o -I .. -I $(BOOST_INC) -m64 -std=c++14 $(CUDA_INC)
MulCusparse.o: MulCusparse.cpp
	$(CXX) -c MulCusparse.cpp -o MulCusparse.o -I .. -I $(BOOST_INC) $(CUDA_INC) -fopenmp -std=c++14 -m64 $(FAST_FLAGS) -O3
lib: MulCusparse.o cuSpmmTest.o cuSpmmOther.o bhsparse.o
	ar rcs libgspgemm.a MulCusparse.o cuSpmmTest.o cuSpmmOther.o bhsparse.o
mult: MulCusparse.o cuSpmmTest.o cuSpmmOther.o bhsparse.o mult.cu
	$(CUDA_CC) $(NVCC_FLAGS) -o mult mult.cu -std=c++14 -m64 -I.. bhsparse.o cuSpmmOther.o cuSpmmTest.o MulCusparse.o -lm -Xcompiler -fopenmp -lcusparse -lcudart -lcublas -lcurand $(BOOST_INC) $(BOOST_LIB) -lboost_timer -lboost_chrono -lboost_system

clean:
	rm -f cuSpmmTest.o
	rm -f cuSpmmOther.o
	rm -f bhsparse.o
	rm -f MulCusparse.o
	rm -f libgspgemm.a

