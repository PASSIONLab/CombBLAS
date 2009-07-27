/** @mainpage Combinatorial BLAS Library (MPI reference implementation)
*
* @authors Aydin Buluc
*
* @section intro Introduction
* This is a reference implementation of the Combinatorial BLAS Library in C++/MPI
* It is purposefully designed for distributed memory platforms though it also runs in uniprocessor and shared-memory (such as multicores) platforms. 
* 
* It contains efficient implementations of novel data structures/algorithms 
* as well as reimplementations of some previously known data structures/algorithms for convenience.
*
* The main data structure is a distributed sparse matrix (SpParMat<IT,NT,DER>) which HAS-A sequential sparse matrix (SpMat<IT,NT>) that 
* can be implemented in various ways as long as it supports the interface of the base class (currently: SpTuples, SpCols, SpDCols)
* For example, the standard way to declare a parallel sparse matrix A that uses 
* - 32-bit integers for indices, 
* - floats for numerical values (nonzeros),
* - SpDCCols<IT,NT> for the underlying sequential matrix operations, 
* - and MPI-2 for communication is:
* 	SpParMPI2<int, float, SpDCCols<int,float> > A;
* The repetitions of int and float types inside the SpDCCols< > is a direct consequence of the static typing of C++
* and is akin to some STL constructs such as vector<int, SomeAllocator<int> >
*
* The supported operation (a growing list) is:
* - Sparse matrix-matrix multiplication on a semiring SR: Mult_AnXBn<SR>(), Mult_AtXBn<SR>(), Mult_AnXBt<SR>(), Mult_AtXBt<SR>()
* - Elementwise multiplication of sparse matrices (A .* B and A .* not(B) in Matlab): EWiseMult()
* - Unary operations on nonzeros: Apply<__unary_op>()
* - Sparse matrix-dense vector multiplication on a semiring
* - Sparse matrix-sparse vector multiplication on a semiring
* - Generalized matrix indexing: operator(const vector<IT> & ri, const vector<IT> & ci)
* - Numeric type conversion through conversion operators
* - Elementwise operations between sparse and dense matrices: EWiseScale() and operator+=()  
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
* - SpThreaded		: shared memory implementation. Uses <a href="http://www.boost.org/doc/html/thread.html"> Boost.Threads</a> for multithreading. 
	\n Uses a logical 2D block decomposition of sparse matrices. 
	\n Asyncronous (to migitate the severe load balancing problem) 
	\n Lock-free (since it relies on the owner computes rule, i.e. \f$ C_{i,j}\f$ is updated by only \f$ P_{i,j} \f$)
* - SpParMPI		: synchronous distributed memory implementation (i.e. executes in \f$p\f$ stages)
	\n Based on SUMMA
	\n Each processor stores its submatrix (block) in SparseDComp format
* - SpParMPI2		: asyncronous distributed memory implementation (i.e. no broadcasting or stages)
	\n If a processor finished its update on \f$ C_{i,j} \f$ using \f$ A_{i,k} \f$, \f$ B_{k,j} \f$, it requests \f$ A_{i,k+1} \f$, \f$ B_{k+1,j}\f$ from their owners right away.
	\n Performs best under MPI-2's passive synchronization and on an interconnect with RDMA support.  

* For internal installation and implementation tricks, consult http://editthis.info/cs240aproject/Main_Page
*
* Test programs demonstrating how to use the library:
* - <a href="http://gauss.cs.ucsb.edu/~aydin/src/TestSeq.cpp"> betwcent.cpp </a> Betwenness centrality computation on directed, unweighted graphs
* - <a href="http://gauss.cs.ucsb.edu/~aydin/src/TestThread.cpp"> mcl.cpp </a> An implementation of the MCL clustering algorithm using Combinatorial BLAS
*
* [1] Aydin Buluç and John R. Gilbert, <it> On the Representation and Multiplication of Hypersparse Matrices </it>. The 22nd IEEE International Parallel and Distributed Processing Symposium (IPDPS 2008), Miami, FL, April 14-18, 2008
*/
