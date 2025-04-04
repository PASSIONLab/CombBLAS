# GALATIC

Sparse Matrix-Sparse Matrix Multiplication CUDA Template library over generalized semirings.

This repository was forked from [AC-SpGEMM](https://github.com/GPUPeople/ACSpGEMM).

This was developed/Tested with
* Linux 4.12
* CUDA compilation tools 11.1
* A V100

---

## Quickstart Guide

### **Orientation**

The headers you likely need for minimal functionality (exclusion of `CSR.cuh` is possible, if you load directly  to/from GPU memory).

```c++
#include "GALATIC/include/CSR.cuh"
#include "GALATIC/include/dCSR.cuh"
#include "GALATIC/include/SemiRingInterface.h"
#include "GALATIC/source/device/Multiply.cuh"
```

Where `CSR.cuh` is used to represent matrix storage in the [Compressed Sparse Row format](https://en.wikipedia.org/wiki/Sparse_matrix) for matrices in CPU memory.  `dCSR.cuh` is the same, but represents data that is stored in GPU/device memory.  

(Note: there exists a `convert` function  in `dCSR.cuh` for converting between the two. The GPU version is required to perform matrix multiplication)

We recommend you look over these files two files, as you will need to construct the input matrices yourself.

Additionally there is a `COO.cuh` for use with the coordinate list format which can be converted to `CSR` (but not `coo` to `dCSR`). The conversion is not particularly optimized.

### **Defining Semirings**

To define your semiring, you statically extend the "abstract" class defined in `SemiRingInterface.h` 
```C++
// SemiRingInterface.h
template <typename T, typename U, typename V>
struct SemiRing {
    typedef T leftInput_t;
    typedef U rightInput_t;
    typedef V output_t;     // Don't worry about these typedefs for now 
    
    V multiply(const T& a, const U& b);
    V add(const V& a, const V& b);

    V AdditiveIdentity();
};
```

Notice that multiplication has a left input type `T`, a right input type `U`, and an output type `V`. Addition has `V` as both an input and an output.

An example follows where multiplication and addition are defined canonically using doubles.

The `__device__` annotation is required.  The `__host__` annotation is needed in if you would like to verify against a CPU SpGEMM implementaiton.

``` c++
// Define Your Semiring
struct Arith_SR : SemiRing<double, double, double>
{
  __host__ __device__ double multiply(const double& a, const double& b) { return a * b; }
  __host__ __device__ double add(const double& a, const double& b)      { return a + b; }
  __host__ __device__ static double AdditiveIdentity()                  { return     0; }
};

```
You may use the "Semiring" structure (e.g. `Arith_SR`) to hold data from outside the matrix (i.e. global device memory) by storing say, a pointer. This will affect performance. 

As to be expected, only memory which is accesible from the GPU is valid. In addition, you should be careful as to not mutate anything such that data races could occur or that an order of operations becomes required.

Use of constructors / destructors is not reccomended for your semiring struct. The destructor for this will be ran multiple times before multiplication is complete. Ideally the Semiring should be [trivally copyable](https://en.cppreference.com/w/cpp/named_req/TriviallyCopyable). Thus you must manually free resources your semiring uses (if any) after you are done.  Additionally, `T`/`U`/`V`  (input / output types) should also be trivially copyable.


### Performing Matrix Multiplication

To decrease the chance of bad error messages, we reccomend using `SEMIRING_TYPE::leftInput_t`, `SEMIRING_TYPE::rightInput_t` and `SEMIRING_TYPE::output_t` for your matrices instead of the literal types of `T` and `U`. This will ensure any type errors occur in your code, rather than the heavily templated library codes.  It will additionally help prevent errors that claim the multiplication function using your parametesr are not found.

```C++
CSR<Arith_SR::leftInput_t> input_A_CPU;
CSR<Arith_SR::rightInput_t> input_B_CPU;

CSR<Arith_SR::output_t> result_mat_CPU;

dCSR<Arith_SR::leftInput_t> input_A_GPU;
dCSR<Arith_SR::rightInput_t> input_B_GPU;

dCSR<Arith_SR::output_t> result_mat_GPU;


/* ...
   ... load data into input_A_CPU, input_B_CPU
   ...*/

// Transfer input matrices onto GPU
// conver out <- in
convert(input_A_GPU, input_A_CPU);
convert(input_B_GPU, input_B_CPU);

// load data into semiring struct. For this one, we don't need to do anything,
// but you still need to pass it in for generality. The cost is trivial.
Arith_SR semiring;


// Setup execution options, we'll skip the details for now.

const int Threads = 256;
const int BlocksPerMP = 1;
const int NNZPerThread = 2;
const int InputElementsPerThreads = 2;
const int RetainElementsPerThreads = 1;
const int MaxChunksToMerge = 16;
const int MaxChunksGeneralizedMerge = 256; // MAX: 865
const int MergePathOptions = 8;


GPUMatrixMatrixMultiplyTraits  DefaultTraits(Threads, BlocksPerMP, NNZPerThread,
                                             InputElementsPerThreads, RetainElementsPerThreads,
                                             MaxChunksToMerge, MaxChunksGeneralizedMerge, MergePathOptions);

const bool Debug_Mode = true;
DefaultTraits.preferLoadBalancing = true;
ExecutionStats stats;
stats.measure_all = false;

// Actually perform the matrix multiplicaiton
ACSpGEMM::Multiply<Arith_SR>(input_A_GPU, input_B_GPU, result_mat_GPU, DefaultTraits, stats, Debug_Mode, semiring);


// load results  onto CPU.
convert(result_mat_GPU, result_mat_GPU);

```

A minimal working example is located in `minimal_example.cu` (note, contains different code currrently).

compile it with

`$ nvcc minimal_example.cu --ftemplate-backtrace-limit 1 --expt-relaxed-constexpr`

Note: `--expt-relaxed-constexpr` is required.


----



### Testing
You can the output against a simple CPU version. (Matrix values, row offsets, column id's).

Simply add the header
```cpp 
#include "GALATIC/include/TestSpGEMM.cuh"
```

and execute	

```cpp
TestSpGEMM(input_A_GPU, input_B_GPU, semiring, [=] (const Arith_SR::output_t &a, const Arith_SR::output_t &b) { return std::abs(a-b) < 0.01; }, DefaultTraits);
```

Default traits is the configuration traits, as 
above. 

The lambda function is function which takes two of your output type, and returns true if they are equivalent, otherwise false. 

Make sure your semiring functions are marked with `__host__`. Addditionally, if you are accessing datastructures outside the matrix, `cudaMallocManaged` is reccomended, as then both the CPU and GPU can access the memory using the same code. 

---
## Important Information


AC-SpGEMM is highly configurable as can be seen with the traits in the `performTestCase`, these traits are implemented as template parameters.
Hence, for all combinations used, the **respective instantiation must be present**.
Instantiations can be created by modifying the call to `Multiply` in `source/GPU/Multiply.cu` in line 781, which is given as
```cpp
bool called = 
	EnumOption<256, 256, 128, // Threads
	EnumOption<3, 4, 1, // BlocksPerMP
	EnumOption<2, 2, 1, // NNZPerThread
	EnumOption<4, 4, 1, // InputElementsPerThreads
	EnumOption<4, 4, 1, // RetainElementsPerThreads
	EnumOption<16, 16, 8, // MaxChunksToMerge
	EnumOption<256, 512, 256, // MaxChunksGeneralizedMerge
	EnumOption<8, 8, 8, // MergePathOptions
	EnumOption<0, 1, 1>>>>>>>>> // DebugMode
			::call(Selection<MultiplyCall<DataType>>(call), scheduling_traits.Threads, scheduling_traits.BlocksPerMp, scheduling_traits.NNZPerThread, scheduling_traits.InputElementsPerThreads, scheduling_traits.RetainElementsPerThreads, scheduling_traits.MaxChunksToMerge, scheduling_traits.MaxChunksGeneralizedMerge, scheduling_traits.MergePathOptions, (int)Debug_Mode);
```
This expanding template will instantiate variants of `MultiplyCall` with the parameters specified in `EnumOption<Start, End, Step>`, so each EnumOption describes all the possible values for a certain property and all different configurations will be instantiated (e.g. BlocksPerMP with `EnumOption<3, 4, 1,` will instantiate the template call with BlocksPerMP=3 and BlocksPerMP=4)

These parameters may require adjusting for optimal performance, or to just run if your semiring is especially large.

---

# About

GALATIC: GPU Accelerated Sparse Matrix Multiplication over Arbitrary 
Semirings (GALATIC) Copyright (c) 2020-2021, The Regents of the 
University of California, through Lawrence Berkeley National Laboratory 
(subject to receipt of any required approvals from the U.S. Dept. of Energy),
Richard Lettich, and GPUPeople. All rights reserved.

If you have questions about your rights to use or distribute this software,
please contact Berkeley Lab's Intellectual Property Office at
IPO@lbl.gov.

NOTICE.  This Software was developed under funding from the U.S. Department
of Energy and the U.S. Government consequently retains certain rights.  As
such, the U.S. Government has been granted for itself and others acting on
its behalf a paid-up, nonexclusive, irrevocable, worldwide license in the
Software to reproduce, distribute copies to the public, prepare derivative 
works, and perform publicly and display publicly, and to permit others to do so.

# FAQ
richardl@berkeley.edu

