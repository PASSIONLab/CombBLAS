# Frequently Asked Questions about Combinatorial BLAS

- [How can I write the output sparse matrices into a human readable file?](#how-can-I-write-the-output-sparse-matrices-into-a-human-readable-file)
- [Does Combinatorial BLAS support in-node multithreading?](#does-combinatorial-blas-support-in-node-multithreading)
- [What input formats do you support?](#what-input-formats-do-you-support)
- [How can I convert a text file into binary so I read it faster later](#how-can-i-convert-a-text-file-into-binary-so-i-read-it-faster-later)
- [Is there a preferred way to prune elements from a SpParMat according to a predicate?](#is-there-a-preferred-way-to-prune-elements-from-a-spparmat-according-to-a-predicate)
- [Does CombBLAS include the API to perform a symmetric permutation on a matrix?](#does-combblas-include-the-api-to-perform-a-symmetric-permutation-on-a-matrix)
- [Does CombBLAS code run on any graph size or there is some limitation on the dimension of the matrix?](#does-combblas-code-run-on-any-graph-size-or-there-is-some-limitation-on-the-dimension-of-the-matrix)
- [What is the difference in implementations of matrix-sparse vector and matrix-dense vector multiply?](#what-is-the-difference-in-implementations-of-matrix-sparse-vector-and-matrix-dense-vector-multiply)
- [How do sparse-matrix based implementations compare with more native implementations?](#how-do-sparse-matrix-based-implementations-compare-with-more-native-implementations)
- [Is there an effort to incorporate the bottom-up BFS of Scott Beamer into CombBLAS?](#is-there-an-effort-to-incorporate-the-bottom-up-bfs-of-scott-beamer-into-combblas)
- [Looking at the output of your Graph500 application, I noticed a large number of self-edges removed. That’s very interesting.](#looking-at-the-output-of-your-graph500-application-i-noticed-a-large-number-of-self-edges-removed-thats-very-interesting)
- [How are you counting the number of edges traversed in Graph500?](#how-are-you-counting-the-number-of-edges-traversed-in-graph500)
- [My computations finish fine but I get an “Attempting to use an MPI routine after finalizing MPICH” afterwards.](#my-computations-finish-fine-but-i-get-an-attempting-to-use-an-mpi-routine-after-finalizing-mpich-afterwards)

## How can I write the output sparse matrices into a human readable file?

`SpParMat::ParallelWriteMM(…)` will create a single file output in human readable coordinate style of the Matrix Market Exchange format. 
There is a similar function for vectors, namely `FullyDistVec::ParallelWrite (…)`. 

 
## Does Combinatorial BLAS support in-node multithreading? 
 
Almost all expensive primitives (SpGEMM, SpMV with sparse vectors, SpMV with dense vectors, EWiseMult, Apply, Set) are hybrid multithreaded within a socket. 
 
## What input formats do you support?

We support three input formats
### 1. [Matrix market exhange format (coordinate)](http://math.nist.gov/MatrixMarket/formats.html) (human readable):

A [matrix market exchange file in coordinate format](http://math.nist.gov/MatrixMarket/formats.html) starts with a mandatory header that specifies the type of the matrix. This is *not to be confused with Coordinate Text File Format* listed under the same link.
The header is followed by several lines of *optional* comments .
A comment begins with "*%*".
After the optional comments, there is another mandatory data header containing three integers denoting the number of rows, columns and nonzero entries in the matrix. 
The rest of the file lists nonzeros of the matrix, one line for each nonzero.
Except for inputs of type *pattern*, each nonzero is represented by three numbers: the row id, the column id and the value of the nonzero. 
For *pattern* inputs, the value field is not present so each non-header line is of two entries.
In contrast to the labeled triples format, matrix market only allows integer vertex identifiers.
This is useful when vertex labels are converted to numeric ids in a preprocessing step to reduce input file size. 
This format is also popular in scientific computing community. A large collection of graphs and sparse matrices are already available in the matrix market format, such as the [The University of Florida Sparse Matrix Collection](https://www.cise.ufl.edu/research/sparse/matrices/).
The same graph shown above with labeled triples format is shown in matrix market format below:

```
%%MatrixMarket matrix coordinate real general
%comments
7	7	12
4	6	0.34
4	2	1.50
6	5	0.67
6	3	1.41
1	7	2.15
2	4	0.55
2	1	0.87
5	7	1.75
7	1	1.4
3	1	0.75
3	2	0.25
3	5	1
```
Here, the header denotes that the file contains a symmetric matrix where the nonzero values are stored in floating point numbers. See the [matrix market](http://math.nist.gov/MatrixMarket/formats.html) website for further detail.

CombBLAS can read this file in parallel as follows:
```cpp
SpParMat <int64_t, double, SpDCCols<int64_t,double> > A;
A.ParallelReadMM(Aname, true, maximum<double>()); 
```

The last parameter is a lambda function that determines what the reader should do when the file includes duplicates. It is optional.

CombBLAS can write a matrix market coordinate file in parallel as follows:
```cpp
A.ParallelWriteMM("A_Output.mtx", true);
```


### 2. Labeled triples format (human readable)

In this format, each line encodes an edge of the graph.
An edge is represented by a triple (source_vertex, destination_vertex, edge_weight). Source and destination vertices are presented by string labels and edge weights are represented by floating point numbers. 
Three fields in a triple is separated by white space.
We show an example for for a graph with seven vertices and 12 edges.  

```
vertex_1	vertex_2	0.34
vertex_1	vertex_4	1.50
vertex_2	vertex_5	0.67
vertex_2	vertex_7	1.41
vertex_3	vertex_6	2.15
vertex_4	vertex_1	0.55
vertex_4	vertex_3	0.87
vertex_5	vertex_6	1.75
vertex_6	vertex_3	1.4
vertex_7	vertex_3	0.75
vertex_7	vertex_4	0.25
vertex_7	vertex_5	1
```

This file can be read by CombBLAS as follows:
```cpp
SpParMat <int64_t, double, SpDCCols<int64_t,double> > B;
FullyDistVec<int64_t, array<char, MAXVERTNAME> > perm = B.ReadGeneralizedTuples(Bname, maximum<double>());
```

A working example can be found here:
https://github.com/PASSIONLab/CombBLAS/blob/master/ReleaseTests/ParIOTest.cpp

Upon completion, `ReadGeneralizedTuples` returns two objects: 
(1) a CombBLAS compliant distributed sparse matrix object, and 
(2) a CombBLAS compliant distributed vector that maps the newly created integer labels {0,...,M-1} and {0,...,N-1} 
into their original string labels so that the program can convert the internal labels back into their original labels for subsequent 
processing or while writing the output. A crucial positive side effect of the `ReadGeneralizedTuples` function is that it automatically 
permutes row and column ids randomly during the relabeling, ensuring load balance of CombBLAS operations that use the resulting distributed 
sparse matrix. For this reason alone, one can use the `ReadGeneralizedTuples` function in lieu of the `ParallelReadMM` function 
if the input is known to be severely load imbalanced. In such cases, reading the input into a distributed sparse matrix `ParallelReadMM` 
and subsequently permuting it within CombBLAS for load balance might not be feasible, because the load imbalance can be high enough 
for some process to run out of local memory before `ParallelReadMM` finishes. 

### 3. Proprietary binary format: 

The binary formatted file starts with a binary header (of size 52 bytes exact) that has the following fields and lengths. 

‘HKDT’: four 8-bit characters describing the beginning of header
Followed by six unsigned 64-bit integers:
* version number
* object size (including the row and column ids)
* format (0: binary, 1: ascii)
* number of rows
* number of columns
* number of nonzeros (nnz)

This is followed by nnz entries, each of which are of size “object size” and parsed by the HANDLER.binaryfill() function supplied by the user. The general signature of the function is:

```cpp
void binaryfill(FILE * rFile, IT & row, IT & col, NT & val)
```

IT is the index template parameter, and NT is the object template parameter. An example is as follows. 

```cpp
template <class IT>
class TwitterReadSaveHandler
{
	void binaryfill(FILE * rFile, IT & row, IT & col, TwitterEdge & val)
	{
			TwitterInteraction twi;
			fread (&twi,sizeof(TwitterInteraction),1,rFile);
			row = twi.from - 1;
			col = twi.to - 1;
			val = TwitterEdge(twi.retweets, twi.follow, twi.twtime); 
	}
}
```

As seen, binaryfill reads indices as well. Please note that the file uses 1-based indices while C/C++ indices are zero based (hence the -1). In general, the number of bits used in the indices by the file should match the number of bits used by the program. If the program’s bits should be larger/smaller; then a cast after the original object creation can be employed. Here is an example to read a file with 64-bit integer indices into 32-bit local -per processor- indices (given that they fit):

```cpp
typedef SpParMat < int64_t, bool, SpDCCols<int64_t,bool> > PSpMat;
typedef SpParMat < int64_t, bool, SpDCCols<int32_t,bool> > PSpMat_s32;
PSpMat A;
A.ReadDistribute(string(argv[2]), 0);
PSpMat_s32 Aeff = PSpMat_s32(A);
```
 
CombBLAS provides default `binaryfill` functions for POD types, as shown here: https://github.com/PASSIONLab/CombBLAS/blob/master/ReleaseTests/Mtx2Bin.cpp


Writing a binary file in parallel is as easy as:
```cpp
A.ParallelBinaryWrite(Bname);
```

This file can then be read in parallel using
```cpp
SpParMat<int64_t, double, SpDCCols<int64_t,double>> B;
B.ReadDistribute(Bname, 0, false, true); // nonum=false: the file has numerical values (i.e., not a pattern only matrix), pario=true: file to be read in parallel
```

## How can I convert a text file into binary so I read it faster later?
You can use the ReleaseTests/Mtx2Bin.cpp example if you have a matrix market text input.
Otherwise, you can create your own converter in just a few lines using the examples in the previous questions. 

## Is there a preferred way to prune elements from a SpParMat according to a predicate?
  Yes, SpParMat::Prune(…) will do it according to a predicate. An overloaded version of the same function, SpParMat::Prune(ri,ci) will prune all entries whose row indices are in ri and column indices are in ci
 
 
## Does CombBLAS include the API to perform a symmetric permutation on a matrix?
  Yes it does. Check out the ReleaseTests/IndexingTiming.cpp for an example.
 

## Does CombBLAS code run on any graph size or there is some limitation on the dimension of the matrix?
 The matrix dimension does not need to be a multiple of sqrt(p) but it should be bigger than sqrt(p). In other words you can have a 5x5 matrix on 4 processors but not on 36 processors. We don't really see the point of using more than |V|^2 processors.

 
## What is the difference in implementations of matrix-sparse vector and matrix-dense vector multiply? 
 
Sparse matrix-sparse vector is "right hand side vector structure" driven. In y = Ax, for each nonzero x_i, we scale the column A(:,i) with that and merge the scaled sparse columns results into y. The computation boils down into merging sparse columns into one. 
 
Sparse matrix-dense vector is slightly different in the sense that it is driven by the matrix structure; you basically stream the matrix. The correctness of both operations are handled by a SPA-like or heap-like data structure that merges multiple intermediate values contributing to the same output location; no atomics are used.
 
 
## How do sparse-matrix based implementations compare with more native implementations
 
Sparse matrix abstraction, like any abstraction, will leave some performance on the table. In particular it is prone to performing extra passes over data or creating extra temporaries (if you've ever programmed in Matlab; this is similar). On the other hand, sparse matrix abstraction gives you "primitives" to implement graph "algorithms" as opposed to the algorithms themselves. For instance, CombBLAS has sparse matrix x sparse vector over a semiring as opposed to BFS, because now using the same primitive one can implement MIS (maximal independent set) too, only by changing the semiring. Or one can perform run time filtering on edges based on the attributes, similarly by changing the semiring functions (therefore extending functionality to semantic graphs). Indeed this is what we've done in our IPDPS'13 paper.
 
 
## Is there an effort to incorporate the bottom-up BFS of Scott Beamer into CombBLAS?
 
Yes, it is already done. Just use the dobfs executable (made from Applications/DirOptBFS.cpp).
 

 
##  Looking at the output of your Graph500 application, I noticed a large number of self-edges removed. That’s very interesting.
 
The duplicate edges problem is inherent to the R-MAT generator on large scale, unless some special kind of noise is added. Check here for a great analysis of this phenomenon: http://arxiv.org/abs/1102.5046
 
---
 
## How are you counting the number of edges traversed in Graph500? 
 
It is calculated by summing the degrees of the discovered vertices using EWiseMult(…) followed by a Reduce(…). Degrees are pre-symmetrization (original edges), so we're not over-counting. However, we count self-loops and duplicates as mentioned in the benchmark specs.
 

 
## My computations finish fine but I get an “Attempting to use an MPI routine after finalizing MPICH” afterwards.
 
To avoid the finalization error, please imitate an example such as ReleaseTests/MultTest.cpp
The curly brackets around the code are intentional. Since distributed objects have MPI related pointers in them, those pointers are released once the destructors are called. In C++, there isn’t a good way to call the destructor manually, so the destructor is called immediately before the program exists, which is after the MPI_Finalize. Since the MPI related objects are destructed after MPI_Finalize, you see this error. Try the curly brackets approach.
 
