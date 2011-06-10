/** @mainpage Combinatorial BLAS Library (MPI reference implementation)
*
* @authors <a href="http://gauss.cs.ucsb.edu/~aydin"> Aydin Buluc </a>, <a href="http://cs.ucsb.edu/~gilbert"> John R. Gilbert </a>
*
* <i> This material is based upon work supported by the National Science Foundation under Grant No. 0709385. Any opinions, findings and conclusions or recommendations expressed in this material are those of the author(s) and do not necessarily reflect the views of the National Science Foundation (NSF) </i>
*
*
* @section intro Introduction
* <b>Download</b> 
* - The latest CMake'd tarball <a href="http://gauss.cs.ucsb.edu/code/CombBLAS/combBLAS_beta_10_cmaked.tar.gz"> here</a>. (NERSC users read <a href="http://gauss.cs.ucsb.edu/code/CombBLAS/NERSC_INSTALL.html">this</a>)
* 	- To create sample applications
* and run simple tests, all you need to do is to execute the following three commands, in the given order, inside the main directory: 
* 		-  <i> cmake . </i>
* 		- <i> make </i>
* 		- <i> ctest -V </i> (you need the testinputs, see below)
* 	- Test inputs are separately downloadable <a href="http://gauss.cs.ucsb.edu/code/CombBLAS/testdata.tar.gz"> here</a>. Extract them inside the PSpGEMM-R1 directory with the command "tar -xzvf testdata.tar.gz"
* - Alternatively (if cmake fails, or you just don't want to install it), you can just imitate the sample makefiles inside the ReleaseTests and Applications 
* directories. Those sample makefiles have the following format: makefile-<i>machine</i>. (example: makefile-neumann) 
* 
* <b>Requirements</b>: You need a recent 
* C++ compiler (g++ version 4.2 or higher - and compatible), a compliant MPI-2 implementation, and a TR1 library (libstdc++ that comes with g++ 
* has them). If not, you can use the boost library and pass the -DNOTR1 option to the compiler (cmake will automatically do it for you); it will work if you just add boost's path to 
* $INCADD in the makefile. The recommended tarball uses the CMake build system, but only to build the documentation and unit-tests, and to automate installation. The chances are that you're not going to use any of our sample applications "as-is", so you can just modify them or imitate their structure to write your own application by just using the header files. There are very few binary libraries to link to, and no configured header files. Like many high-performance C++ libraries, the Combinatorial BLAS is mostly templated. 
* 
* <b>Documentation</b>:
* This is a reference implementation of the Combinatorial BLAS Library in C++/MPI.
* It is purposefully designed for distributed memory platforms though it also runs in uniprocessor and shared-memory (such as multicores) platforms. 
* It contains efficient implementations of novel data structures/algorithms
* as well as reimplementations of some previously known data structures/algorithms for convenience. More details can be found in the accompanying paper [1].
*
* The main data structure is a distributed sparse matrix ( SpParMat <IT,NT,DER> ) which HAS-A sequential sparse matrix ( SpMat <IT,NT> ) that 
* can be implemented in various ways as long as it supports the interface of the base class (currently: SpTuples, SpCCols, SpDCCols).
*
* For example, the standard way to declare a parallel sparse matrix A that uses 32-bit integers for indices, floats for numerical values (nonzeros),
* SpDCCols <int,float> for the underlying sequential matrix operations is: 
* - SpParMat<int, float, SpDCCols<int,float> > A; 
*
* The repetitions of int and float types inside the SpDCCols< > is a direct consequence of the static typing of C++
* and is akin to some STL constructs such as vector<int, SomeAllocator<int> >
*
* Sparse and dense vectors can be distributed either along the diagonal processor or to all processor. The latter is more space efficient and provides 
* much better load balance for SpMSV (sparse matrix-sparse vector multiplication) but the former is simpler and perhaps faster for SpMV 
* (sparse matrix-dense vector multiplication) 
*
*
* The supported operations (a growing list) are:
* - Sparse matrix-matrix multiplication on a semiring SR: Mult_AnXBn_Synch(), and other variants
* - Elementwise multiplication of sparse matrices (A .* B and A .* not(B) in Matlab): EWiseMult()
* - Unary operations on nonzeros: SpParMat::Apply()
* - Matrix-matrix and matrix-vector scaling (the latter scales each row/column with the same scalar of the vector) 
* - Reductions along row/column: SpParMat::Reduce()
* - Sparse matrix-dense vector multiplication on a semiring, SpMV()
* - Sparse matrix-sparse vector multiplication on a semiring, SpMV()
* - Generalized matrix indexing: operator(const vector<IT> & ri, const vector<IT> & ci)
* - Numeric type conversion through conversion operators
* - Elementwise operations between sparse and dense matrices: SpParMat::EWiseScale() and operator+=()  
* 
* All the binary operations can be performed on matrices with different numerical value representations.
* The type-traits mechanism will take care of the automatic type promotion, and automatic MPI data type determination.
* Of course, you have to declare the return value type appropriately (until C++0x is out, which has <a href="http://www.research.att.com/~bs/C++0xFAQ.html#auto"> auto </a>) 
*
* Some features it uses:
* - templates (for generic types, and for metaprogramming through "curiously recurring template pattern")
* - operator overloading 
* - compositors (to avoid intermediate copying) 
* - <a href="http://www.boost.org"> boost library </a> or TR1 whenever necessary (mostly for shared_ptr and tuple) 
* - standard library whenever possible
* - Reference counting using shared_ptr for IITO (implemented in terms of) relationships
* - MPI-2 one-sided operations
* - As external code, it utilizes 
*	- sequence heaps of <a href="http://www.mpi-inf.mpg.de/~sanders/programs/"> Peter Sanders </a>.
*	- a modified (memory efficient) version of the Viral Shah's <a href="http://gauss.cs.ucsb.edu/~viral/PAPERS/psort/html/psort.html"> PSort </a>.
*	- a modified version of the R-MAT generator from <a href="http://graph500.org"> Graph 500 reference implementation </a>
* 
* Important Sequential classes:
* - SpTuples		: uses triples format to store matrices, mostly used for input/output and intermediate tasks (such as sorting)
* - SpCCols		: multiplication is similar to Matlab's, holds CSC. 
* - SpDCCols		: implements Alg 1B and Alg 2 [2], holds DCSC.

* Important Parallel classes:
* - SpParMat		: distributed memory MPI implementation 
	\n Each processor locally stores its submatrix (block) as a sequential SpDCCols object
	\n Uses a polyalgorithm for SpGEMM. 
	\n If robust MPI-2 support is not available, then it reverts back to a less scalable synchronous algorithm that is based on SUMMA [3]
	\n Otherwise, it uses an asyncronous algorithm based on one sided communication. This performs best on an interconnect with RDMA support
* - FullyDistVec	: dense vector distributed to all processors
* - FullyDistSpVec:	: sparse vector distributed to all processors
*
* 
* <b> Applications </b>  implemented using Combinatorial BLAS:
* - BetwCent.cpp : Betweenness centrality computation on directed, unweighted graphs. Download sample input <a href=" http://gauss.cs.ucsb.edu/code/CombBLAS/scale17_bc_inp.tar.gz"> here </a>.
* - MCL.cpp : An implementation of the MCL graph clustering algorithm.
* - Graph500.cpp: A conformant implementation of the <a href="http://graph500.org">Graph 500 benchmark</a>.
* 
* <b> Performance </b> results of the first two applications can be found in the design paper [1]; Graph 500 results are in a recent BFS paper [4]
*
* Test programs demonstrating how to use the library:
* - TransposeTest.cpp : File I/O and parallel transpose tests
* - MultTiming.cpp : Parallel SpGEMM tests
* - IndexingTest.cpp: Various sparse matrix indexing usages
* - FindSparse.cpp : Parallel find/sparse routines akin to Matlab's
*
* <b> Citation: </b> Please cite the design paper [1] if you end up using the Combinatorial BLAS in your research.
*
* - [1] Aydin Buluc and John R. Gilbert, <i> The Combinatorial BLAS: Design, implementation, and applications </i>. International Journal of High Performance Computing Applications (IJHPCA), to appear. <a href="http://www.cs.ucsb.edu/research/tech_reports/reports/2010-18.pdf"> Preprint </a>
* - [2] Aydin Buluc and John R. Gilbert, <i> On the Representation and Multiplication of Hypersparse Matrices </i>. The 22nd IEEE International Parallel and Distributed Processing Symposium (IPDPS 2008), Miami, FL, April 14-18, 2008
* - [3] Aydin Buluc and John R. Gilbert, <i> Challenges and Advances in Parallel Sparse Matrix-Matrix Multiplication </i>. The 37th International Conference on Parallel Processing (ICPP 2008), Portland, Oregon, USA, 2008
* - [4] Aydin Buluc and Kamesh Madduri, <i> Parallel Breadth-First Search on Distributed-Memory Systems </i>. Preprint available at ArXiv.org
* - [5] Aydin Buluc. <i> Linear Algebraic Primitives for Computation on Large Graphs </i>. PhD thesis, University of California, Santa Barbara, 2010. <a href="http://gauss.cs.ucsb.edu/~aydin/Buluc_Dissertation.pdf"> PDF </a>
* 
*/
