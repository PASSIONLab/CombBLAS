# Frequently Asked Questions about Combinatorial BLAS

- [How can I write the output sparse matrices into a file?](#how-can-I-write-the-output-sparse-matrices-into-a-human-readable-file)
- [Does Combinatorial BLAS support in-node multithreading?](#does-combinatorial-blas-support-in-node-multithreading)

## How can I write the output sparse matrices into a human readable file?

`SpParMat::ParallelWriteMM(…)` will create a single file output in human readable coordinate style of the Matrix Market Exchange format. 
There is a similar function for vectors, namely `FullyDistVec::ParallelWrite (…)`. 

 
## Does Combinatorial BLAS support in-node multithreading? 
 
Almost all expensive primitives (SpGEMM, SpMV with sparse vectors, SpMV with dense vectors, EWiseMult, Apply, Set) are hybrid multithreaded within a socket. Read this example.
 
## What input formats do you support?

We support three input formats
1. [Matrix market coordinate format](http://math.nist.gov/MatrixMarket/formats.html) (human readable):

A [matrix market](http://math.nist.gov/MatrixMarket/formats.html) file starts with a mandatory header that specifies the type of the matrix. 
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


2. Labeled triples format (human readable)

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
(2) a CombBLAS compliant distributed vector that maps the newly created integer labels {0,...,M-1} and {0,...,N-1}$ 
into their original string labels so that the program can convert the internal labels back into their original labels for subsequent 
processing or while writing the output. A crucial positive side effect of the `ReadGeneralizedTuples` function is that it automatically 
permutes row and column ids randomly during the relabeling, ensuring load balance of CombBLAS operations that use the resulting distributed 
sparse matrix. For this reason alone, one can use the `ReadGeneralizedTuples` function in lieu of the `ParallelReadMM` function 
if the input is known to be severely load imbalanced. In such cases, reading the input into a distributed sparse matrix `ParallelReadMM` 
and subsequently permuting it within CombBLAS for load balance might not be feasible, because the load imbalance can be high enough 
for some process to run out of local memory before `ParallelReadMM` finishes. 

3. Proprietary binary format:

The binary formatted file starts with a binary header (of size 52 bytes exact) that has the following fields and lengths. 

‘HKDT’: four 8-bit characters describing the beginning of header
Followed by six unsigned 64-bit integers:
**	version number
**	object size (including the row and column ids)
**	format (0: binary, 1: ascii)
**	number of rows
**	number of columns
**	number of nonzeros (nnz)

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

## How can I convert a text file into binary so I read it faster?
You can use the Mtx2Bin example if you have a matrix market text input.
Otherwise, you can create your own converter in just a few lines using the examples in the previous questions. 

## Is there a preferred way to prune elements from a SpParMat according to a predicate?
 
A4: Yes, SpParMat::Prune(…) will do it according to a predicate. An overloaded version of the same function, SpParMat::Prune(ri,ci) will prune all entries whose row indices are in ri and column indices are in ci
 
---
 
Q5: I am trying to run CombBLAS on Windows but the MS MPI does not seem to support MPI C++ bindings.
 
A5: Combinatorial BLAS recently (version 1.3.0) switched to C-API for all its internal MPI calls. After that, we've also compiled CombLAS on a windows machine without problems. However, we recommend using an open source MPI for windows too, such as MPICH-2.
 
---
 
Q6: I would like to use Combinatorial BLAS for some parallel sparse matrix-matrix multiplications. This works quite well, however when I try to assign a m x 1 sparse matrix (actually a vector) to the first column of an existing sparse matrix with SpAsgn I get an error saying: "Matrix is too small to be splitted". Is this because it's not possible to use SpAsgn on vector-like matrices?
 
A6: SpAsgn internally uses a memory efficient Mult_AnXBn_DoubleBuff as opposed to Mult_AnXBn_Synch). You might probably go into SpParMat<IT,NT,DER>::SpAsgn(...) and change occuranges of Mult_AnXBn_DoubleBuff to Mult_AnXBn_Synch. However, this will likely only solve your problem for the serial case. because ComBBLAS can not effectively 2D decompose an m x 1 matrix: each dimension should ideally be at least sqrt(p). It is much better to represent that vector as a FullyDistSpVec.
 
---
 
Q7: Starting from a general sparse matrix Z, I want to construct the symmetric matrix M: [[X'X; X'Z];[Z'X; Z'Z]], where X is a vector of 1's. Thus the element at position (1,1) is simply the number of columns of Z, and the vector Z'X contains the sums per column of Z. For now, I have a working code, but it is quite sloppy because I do not find a function for which I can easily increase the dimension of a sparse matrix or even set an element to a specific value. Is there any function in Combinatorial BLAS which can do this? 
 
A7: Not out of the box. You don't want an SpAsgn or any variant of it because it can't grow the matrix. You want some sort of matrix append. How about using Find(…) and Sparse(…)?  The Matlab care of what you want to do is:
 
X = ones(size(Z,2),1) 
M = [X' * X, X' * Z; Z'* X, Z' * Z]
 
Supporting such a general concatenation efficiently might be hard to add at this point. Instead,  there is a Concatenate(…) function for vectors. Armed with Concatenate(…), find(), and the sparse-like constructor, one can solve your problem.  Check out the working example in ReleaseTests/FindSparse.cpp
 
---
 
Q8: Does CombBLAS include the API to perform a symmetric permutation on a matrix, as explained in your SISC paper? 
 
A8: Yes it does. Check out the ReleaseTests/IndexingTiming.cpp for an example.
 
--- 
 
Q9: How can I use small test case to see whether the operation on matrix is correct? In other words, how do I print all the information of a matrix with each value in matrix? 
I can use PrintInfo to print basic information, but it only gives me number of rows and columns and nnz
 
A9: Our recommendation is to use SaveGathered(…) to dump the whole matrix into a file in triples (matrix market) format. For vectors, we have a much much faster version:  FullyDistVec::ParallelWrite (…)
 
---
 
Q10: Does CombBLAS code run on any graph size or there is some limitation on the dimension of the matrix A. I mean should it be a multiple of sqrt(p) where p is total number of processors. 
 
A10: No, the matrix dimension does not have to be a multiple of sqrt(p) but it should be bigger than sqrt(p). In other words you can have a 5x5 matrix on 4 processors but not on 36 processors. We don't really see the point of using more than |V|^2 processors.
 
---
 
Q11: My comparison results on real graph inputs revealed something weird. In input loc-gowalla, how can 16 processors time(called time_16) and 
64 processors time(called time_64) which time_64*4<time_16  which is more than linear scale? 
 
A11: The complexity of the parallel algorithm drops as sub-matrices owned by each processor gets sparser. In particular, it is proportional to O(flops x log(ni)) where ni is the size of the intersection of the set of nonzero columns of Aik and nonzero rows of Bkj for A*B. What might happen as p increases is that there is a phase transition that makes ni drop significantly for your input (for p=64, each sub-matrix will have only ~1.2 nonzeros per row or column). More details are in the SISC paper and the references therein. Hope this makes sense. This is why I don't suggest people use CombBLAS for small p (< 40) because it is not on the top of its game for small number of processors. 
 
--- 
 
Q12: Should the input file have nodes numbered from 1 or it is fine if the nodes are numbered from 0?
 
A12: If you're using the human readable matrix market format as your input, then it should be 1-indexed. 
 
---
 
Q13: I'm wondering for breadth-first-search, under the hood does the matrix-vector multiplication method change based on the sparsity of the frontier vector, or does the underlying matrix-vector multiplication assume the frontier is always sparse?
 
A13: Depending on your definition of sparseness, the frontier is almost always sparse. We use the pragmatic definition of "sparse" in the sense that a vector is sparse if it is worth taking advantage of the sparsity in there. I'd guess, for a dense vector assumption to be competitive, it would have to have at least 1/3 of its potential locations nonzero. However, I might be wrong (and you're welcome to prove me wrong). To answer your question more directly, CombBLAS supports both dense and sparse right hand side vectors, but the specific BFS implementation does not adapt. 
 
---
 
Q14: Could you briefly explain the difference in your implementations of matrix-sparse vector and matrix-dense vector multiply? For example, is the sparse vector case a write-based approach: Every element updates all of its neighbors (from a graph-theoretic standpoint) locations in the output vector; and the dense vector case a read-based approach: Every element reads some value from each of its neighbors and updates its own entry in the resulting vector?
 
A14: Sparse matrix-sparse vector is "right hand side vector structure" driven. In y = A*x, for each nonzero x_i, we scale the column A(:,i) with that and merge the scaled sparse columns results into y. The computation boils down into merging sparse columns into one. Combinatorial BLAS is a matrix-vector based library, so thinking in terms of updates on single entries is probably not the right abstraction.
 
Sparse matrix-dense vector is slightly different in the sense that it is driven by the matrix structure; you basically stream the matrix. The correctness of both operations are handled by a SPA-like or heap-like data structure that merges multiple intermediate values contributing to the same output location; no atomics are used.
 
--- 
 
Q15: I would like to get your opinion on how sparse-matrix based implementations compare with more native implementations
 
A15: Sparse matrix abstraction, like any abstraction, will leave some performance on the table. In particular it is prone to performing extra passes over data or creating extra temporaries (if you've ever programmed in Matlab; this is similar). On the other hand, sparse matrix abstraction gives you "primitives" to implement graph "algorithms" as opposed to the algorithms themselves. For instance, CombBLAS has sparse matrix x sparse vector over a semiring as opposed to BFS, because now using the same primitive one can implement MIS (maximal independent set) too, only by changing the semiring. Or one can perform run time filtering on edges based on the attributes, similarly by changing the semiring functions (therefore extending functionality to semantic graphs). Indeed this is what we've done in our upcoming IPDPS'13 paper.
 
---
 
Q16: Is there an effort to incorporate the bottom-up BFS of Scott Beamer into CombBLAS?
 
A16: Yes, it is already done. Just use the dobfs executable (made from DirOptBFS.cpp).
 
---
 
Q17: My serial code is faster than CombBLAS on a single core.
 
A17: I believe that. CombBLAS targets "scalability", not optimizing the single core performance.
 
Examples:
- think about the 2D BFS. CombBLAS does not use a CSR like data structure because that is not memory scalable due to problems of hypersparsity in large concurrencies. Instead CombBLAS opts to use a slower (about 2x around 1000 cores) but memory scalable format called DCSC.  
- think about betweenness centrality which uses sparse matrix-matrix multiply. CombBLAS doesn't use the fastest serial algorithm as its subroutine because it doesn't  scale to thousands of cores. Instead it uses a outer-product algorithm that is significantly slower for p=1, but scales indefinitely.
 
--- 
 
Q18: Looking at the output of your Graph500 application, I noticed a large number of self-edges removed. That’s very interesting.
 
A18: The duplicate edges problem is inherent to the R-MAT generator on large scale, unless some special kind of noise is added. Check here for a great analysis of this phenomenon: http://arxiv.org/abs/1102.5046
 
---
 
Q19: How are you counting the number of edges traversed in Graph500? Is this still using the original verify.c file provided with the reference version of the Graph500 benchmark and passing in the parent tree?
 
A19: It is calculated by summing the degrees of the discovered vertices using EWiseMult(…) followed by a Reduce(…). Degrees are pre-symmetrization (original edges), so we're not over-counting. However, we count self-loops and duplicates as mentioned in the benchmark specs.
 
---
 
Q20: My computations finishes fine but I get an “Attempting to use an MPI routine after finalizing MPICH” afterwards.
 
A20: To avoid the finalization error, please imitate an example such as MultTest.cpp: http://gauss.cs.ucsb.edu/~aydin/CombBLAS/html/_mult_test_8cpp_source.html
The curly brackets around the code are intentional. Since distributed objects have MPI related pointers in them, those pointers are released once the destructors are called. In C++ (at least until C++11) there isn’t a good way to call the destructor manually, so the destructor is called immediately before the program exists, which is after the MPI_Finalize. Since the MPI related objects are destructed after MPI_Finalize, you see this error. Try the curly brackets approach.
 
Go back to the the Combinatorial BLAS home page.
