## Example GNU Makefile for using UPC with CUDA
## Y. Zheng
## LBNL 2009
## 

CUDA_INSTALL_PATH ?= /usr/local/cuda
INCLUDE := -I$(CUDATOOLKIT_HOME)/inc


#CUBIN_ARCH_FLAG := -m32
CUDA_LIB64 := -L$(CUDATOOLKIT_HOME)/lib64
CUDA_LIBS := $(CUDA_LIB64) -lcudart #-L$(CUDATOOLKIT_HOME)/lib -lcutil_x86_64
NVCC := nvcc
NVCCFLAGS := $(INCLUDE)

# C++ compiler
CXX := mpicxx
CXXFLAGS := -std=c++0x -O2 -fpermissive -DPARALLELREDUCE -DDEBUG #-DBUPC

# C compiler
CC := mpicc
CFLAGS := -Wall -O2

# UPC compiler
# UPCC := upcc
# UPCC := upcc -network=smp -pthreads
#UPCC := upcc -network=ibv
UPCC := upcc -network=gemini
# UPCC := cc -h upc
UPCC_FLAGS := $(INCLUDE) --uses-mpi -DBUPC_USE_UPC_NAMESPACE -v 

# linker
LINK := $(UPCC) #-link-with=nvcc 
LFLAGS := --uses-mpi -link-with=$(CXX) -DBUPC_USE_UPC_NAMESPACE -DBUPC

COMBBLAS = ..

$(COMBBLAS)/graph500-1.2/generator/libgraph_generator_seq.a:
	$(MAKE) -C $(COMBBLAS)/graph500-1.2/generator


# Rules of compiling different source file types
%.o : %.upc
	$(UPCC) $(UPCC_FLAGS) -o $@ -c $<

%.o : %.c 
	$(CC) $(CFLAGS) -o $@ -c $<

%.o : %.cpp 
	$(CXX) $(CXXFLAGS) -o $@ -c $<

%.o : %.cxx 
	$(CXX) $(CXXFLAGS) -o $@ -c $<

%.o : %.cu 
	$(NVCC) $(NVCCFLAGS) -o $@ -c $<

%.cubin : %.cu 
	$(NVCC) $(NVCCFLAGS) -o $@ -cubin $<

%.ptx : %.cu
	$(NVCC)  $(NVCCFLAGS) -o $@ -ptx $<

all: hello pspgemm

clean:
	rm -rf hello hello_pthread-link pspgemm
	rm -f *.o
	rm -rf ../*.o
	rm -f ./graph500-1.2/generator/*.o
	rm -f ./graph500-1.2/generator/libgraph_generator_seq.a

hello: hello_cuda.o gpu_data.o hello.o hypersparsegemm.o
	$(LINK) $(LFLAGS) -o $@ $^ $(CUDA_LIBS)

pspgemm: pspgemm.o hypersparsegemm.o GenRMat.o $(COMBBLAS)/MPIType.o $(COMBBLAS)/graph500-1.2/generator/libgraph_generator_seq.a $(COMBBLAS)/CommGrid.o Cessentials.o 
	$(LINK) $(LFLAGS) -o $@ $^ -L$(COMBBLAS)/graph500-1.2/generator -lgraph_generator_seq

simple: simple.o GenRMatSimple.o $(COMBBLAS)/MPIType.o 
	$(LINK) $(LFLAGS) -o $@ $^

mpipspgemm: mpipspgemm.o GenRMat.o $(COMBBLAS)/MPIType.o $(COMBBLAS)/graph500-1.2/generator/libgraph_generator_seq.a $(COMBBLAS)/CommGrid.o Cessentials.o SUMMALayer.o
	$(CXX) $(CXXFLAGS) -o $@ $^ -L$(COMBBLAS)/graph500-1.2/generator -lgraph_generator_seq

pingpong: gpu_data.o pingpong.o
	$(LINK) $(LFLAGS) -o $@ $^ $(LIBS)

