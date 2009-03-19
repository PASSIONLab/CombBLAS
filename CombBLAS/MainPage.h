/** @mainpage Sequential/Parallel SpGEMM Library
*
* @authors Aydin Buluc
*
* @section intro Introduction
* This is a Sparse Matrix Multiplication Library 
* It targets uniprocessor, shared-memory (such as multicores), and especially distributed memory platforms. 
* 
* It contains efficient implementations of novel algorithms 
* as well as reimplementations of some previously known algorithms (for comparison).
*
* It is written in C++ using:
* - templates (for generic types, and for metaprogramming through "curiously recurring template pattern")
* - operator overloading 
* - compositors (to avoid intermediate copying) 
* - <a href="http://www.boost.org"> boost library </a> whenever necessary 
* - standard library whenever possible
* - Reference counting using shared_ptr for IITO (implemented in terms of) relationships
* - MPI-2 one-sided operations
* - As external code, it utilizes sequence heaps of Peter Sanders http://www.mpi-inf.mpg.de/~sanders/papers/spqjea.ps.gz.
*
* 
* Sequential classes:
* - SparseTriplets	: uses triples format to store matrices, only used for input/output and intermediate tasks (such as sorting)
	Does not contain a working multiplication routine. Only a stub exists 
* - SparseColumn	: multiplication is similar to Matlab's, holds CSC. 
* - SparseDColumn	: implements Alg 1B and Alg 2 [1] (depending on the macro in src/SpDefines.h), holds DCSC.

* Parallel classes:
* - SparseThreaded	: shared memory implementation. Uses <a href="http://www.boost.org/doc/html/thread.html"> Boost.Threads</a> for multithreading. 
	\n Uses a logical 2D block decomposition of sparse matrices. 
	\n Asyncronous (to migitate the severe load balancing problem) 
	\n Lock-free (since it relies on the owner computes rule, i.e. \f$ C_{i,j}\f$ is updated by only \f$ P_{i,j} \f$)
* - SparsePar2D		: synchronous distributed memory implementation (i.e. executes in \f$p\f$ stages)
	\n Based on SUMMA
	\n Each processor stores its submatrix (block) in SparseDComp format
	\n Uses <a href="http://www.osl.iu.edu/~dgregor/boost.mpi/doc/"> Boost.MPI</a> for communication
	\n Slightly more scalable but much slower (a factor of 2) than SparseSumSync. 
* - SparseSumSync	: similar to SparseSumSync in principle, but uses SparseDColumn for storing submatrices.
* - SparseSumAsync	: asyncronous distributed memory implementation (i.e. no broadcasting or stages)
	\n If a processor finished its update on \f$ C_{i,j} \f$ using \f$ A_{i,k} \f$, \f$ B_{k,j} \f$, it requests \f$ A_{i,k+1} \f$, \f$ B_{k+1,j}\f$ from their owners right away.
	\n A ServerThread is used for serving send_matrix requests to provide asyncronous computation.
	\n Threading with 2 threads per processor (1 for compute, 1 for communicate) severely degrades the performance.
	\n OpenMPI does not really work with multithreaded environments. 
	\n MPICH2 does work but it uses polling to minimize latency (no thread_yield), therefore even more degrading the performance.  
* - SparseSumGas	: asyncronous distributed memory implementation using <a href="http://gasnet.cs.berkeley.edu/"> GASNET </a>.

* For internal installation and implementation tricks, consult http://editthis.info/cs240aproject/Main_Page
*
* Test programs demonstrating how to use the library:
* - <a href="http://gauss.cs.ucsb.edu/~aydin/src/TestSeq.cpp"> TestSeq.cpp </a> for SparseDComp and SparseDColumn usage
* - <a href="http://gauss.cs.ucsb.edu/~aydin/src/TestThread.cpp"> TestThread.cpp </a> for SparseThreaded usage
* - <a href="http://gauss.cs.ucsb.edu/~aydin/src/TestPar.cpp"> TestPar.cpp </a> for SparsePar2D, SparseSumSync, and SparseSumAsych usage
* - <a href="http://gauss.cs.ucsb.edu/~aydin/src/TestGas.cpp"> TestGas.cpp </a> for SparseSumGas usage

* Some important notes to users:
* - It is the programmer's responsibility to make sure that A and B uses the same implementation as C,
	\n because PSpGEMM does not support operations between different sparse matrix implementations.
	\n For example SparseDColumn<T> C = A * B; works only if A and B are also implemented as SparseDColumn
	\n This is more subtle to ensure in compile time in the presence of polymorphism. If you define A, B, and C as pointers to SparseMatrix, then make sure all uses the same implementation.
* - PSpGEMM *WILL* support mixed precision arithmetic (An object of type SparseMatrix<T1> can be multiplied with SparseMatrix<T2> for T1 != T2)
*
* [1] Aydin Buluç and John R. Gilbert, <it> On the Representation and Multiplication of Hypersparse Matrices </it>. The 22nd IEEE International Parallel and Distributed Processing Symposium (IPDPS 2008), Miami, FL, April 14-18, 2008
*/
