/** @mainpage Combinatorial BLAS Library (MPI reference implementation)
*
* @authors Aydin Buluc
*
* @section intro Introduction
* Download the latest tarball <a href="http://gauss.cs.ucsb.edu/code/CombBLAS/combinatorialBLAS-alpha_10.tar.gz"> here </a>. You need a recent 
* C++ compiler (g++ version 4.2 or higher - and compatible), a compliant MPI-2 implementation, and a TR1 library (libstdc++ that comes with g++ 
* has them). If not, you can use the boost library and pass the -DNOTR1 option to the compiler; it will work if you just add boost's path to 
* $INCADD in the makefile. 
* 
* This is a reference implementation of the Combinatorial BLAS Library in C++/MPI.
* It is purposefully designed for distributed memory platforms though it also runs in uniprocessor and shared-memory (such as multicores) platforms. 
* It contains efficient implementations of novel data structures/algorithms
* as well as reimplementations of some previously known data structures/algorithms for convenience.
*
* The main data structure is a distributed sparse matrix ( SpParMat <IT,NT,DER> ) which HAS-A sequential sparse matrix ( SpMat <IT,NT> ) that 
* can be implemented in various ways as long as it supports the interface of the base class (currently: SpTuples, SpCCols, SpDCCols)
* For example, the standard way to declare a parallel sparse matrix A that uses 
* - 32-bit integers for indices, 
* - floats for numerical values (nonzeros),
* - SpDCCols <int,float> for the underlying sequential matrix operations, 
* - and MPI-2 for communication is:
* 	SpParMat<int, float, SpDCCols<int,float> > A;
* The repetitions of int and float types inside the SpDCCols< > is a direct consequence of the static typing of C++
* and is akin to some STL constructs such as vector<int, SomeAllocator<int> >
*
* The supported operation (a growing list) is:
* - Sparse matrix-matrix multiplication on a semiring SR: Mult_AnXBn_Synch(), and other variants
* - Elementwise multiplication of sparse matrices (A .* B and A .* not(B) in Matlab): EWiseMult()
* - Unary operations on nonzeros: SpParMat::Apply()
* - Matrix-matrix and matrix-vector scaling (the latter scales each row/column with the same scalar of the vector) 
* - Reductions along row/column: SpParMat::Reduce()
* - Sparse matrix-dense vector multiplication on a semiring
* - Sparse matrix-sparse vector multiplication on a semiring
* - Generalized matrix indexing: operator(const vector<IT> & ri, const vector<IT> & ci)
* - Numeric type conversion through conversion operators
* - Elementwise operations between sparse and dense matrices: SpParMat::EWiseScale() and operator+=()  
* 
* All the binary operations can be performed on matrices with different numerical value representations.
* The type-traits mechanism will take care of the automatic type promotion.
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
* - As external code, it utilizes sequence heaps of <a href="http://www.mpi-inf.mpg.de/~sanders/programs/"> Peter Sanders </a>.
*
* 
* Sequential classes:
* - SpTuples		: uses triples format to store matrices, mostly used for input/output and intermediate tasks (such as sorting)
* - SpCCols		: multiplication is similar to Matlab's, holds CSC. 
* - SpDCCols		: implements Alg 1B and Alg 2 [1], holds DCSC.

* Parallel classes:
* - SpParMat		: distributed memory MPI implementation 
	\n Each processor locally stores its submatrix (block) as a sequential SpDCCols object
	\n Uses a polyalgorithm for SpGEMM. 
	\n If robust MPI-2 support is not available, then it reverts back to a less scalable synchronous algorithm that is based on SUMMA [2]
	\n Otherwise, it uses an asyncronous algorithm based on one sided communication. This performs best on an interconnect with RDMA support
* - SpThreaded (not included)	: shared memory implementation. Uses <a href="http://www.boost.org/doc/html/thread.html"> Boost.Threads</a> for multithreading. 
	\n Uses a logical 2D block decomposition of sparse matrices. 
	\n Asyncronous (to migitate the severe load balancing problem) 
	\n Lock-free (since it relies on the owner computes rule, i.e. C_{ij} is updated by only P_{ij})

* For internal installation and implementation tricks, consult http://editthis.info/cs240aproject/Main_Page
*
* Applications implemented using Combinatorial BLAS:
* - <a href="http://gauss.cs.ucsb.edu/~aydin/src/BetwCent.cpp"> BetwCent.cpp </a> Betwenness centrality computation on directed, unweighted graphs. Download sample input <a href=" http://gauss.cs.ucsb.edu/code/CombBLAS/SCALE17BTW-TRANSBOOL/input.txt"> here </a>
* - <a href="http://gauss.cs.ucsb.edu/~aydin/src/MCL.cpp"> MCL.cpp </a> An implementation of the MCL graph clustering algorithm 
*
* Test programs demonstrating how to use the library:
* - <a href="http://gauss.cs.ucsb.edu/~aydin/src/TransposeTest.cpp"> TransposeTest.cpp </a> File I/O and parallel transpose tests
* - <a href="http://gauss.cs.ucsb.edu/~aydin/src/MultTiming.cpp"> MultTiming.cpp </a> Parallel SpGEMM tests
*
* [1] Aydin Buluc and John R. Gilbert, <it> On the Representation and Multiplication of Hypersparse Matrices </it>. The 22nd IEEE International Parallel and Distributed Processing Symposium (IPDPS 2008), Miami, FL, April 14-18, 2008
*
* [2] Aydin Buluc and John R. Gilbert, <it> Challenges and Advances in Parallel Sparse Matrix-Matrix Multiplication </it>. The 37th International Conference on Parallel Processing (ICPP 2008), Portland, Oregon, USA, 2008
*/
