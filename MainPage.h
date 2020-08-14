/** @mainpage The Combinatorial BLAS Library
*
* @authors <a href="http://crd.lbl.gov/departments/computer-science/PAR/staff/ariful-azad/"> Ariful Azad </a>, <a href="http://eecs.berkeley.edu/~aydin"> Aydın Buluç </a>, and <a href="http://cs.ucsb.edu/~gilbert"> John R. Gilbert </a> (with contributions from <a href="http://www.cs.ucsb.edu/~alugowski/">Adam Lugowski</a>, <a href="http://www.cs.berkeley.edu/~sbeamer/">Scott Beamer</a> and <a href="https://ucsb-ccs-computing.github.io/alumni/konolige_tristan/">Tristan Konolige</a>).
*
* @copyright
* <i> 

Combinatorial BLAS, Copyright (c) 2018, The Regents of the University of California, through Lawrence Berkeley National Laboratory (subject to receipt of any required approvals from the U.S. Dept. of Energy) and University of California, Santa Barbara.  All rights reserved.

If you have questions about your rights to use or distribute this software, please contact Berkeley Lab's Innovation & Partnerships Office at  IPO@lbl.gov.
 

NOTICE.  This Software was developed under funding from the U.S. Department of Energy and the U.S. Government consequently retains certain rights. As such, the U.S. Government has been granted for itself and others acting on its behalf a paid-up, nonexclusive, irrevocable, worldwide license in the Software to reproduce, distribute copies to the public, prepare derivative works, and perform publicly and display publicly, and to permit other to do so. 

This material is based upon work supported by the National Science Foundation under Grant No. 0709385 and by the Department of Energy, Office of Science, ASCR Contract No. DE-AC02-05CH11231. Any opinions, findings and conclusions or recommendations expressed in this material are those of the author(s) and do not necessarily reflect the views of the National Science Foundation (NSF) and the Department of Energy (DOE). This software is released under the following <a href="http://eecs.berkeley.edu/~aydin/CombBLAS_FILES/LICENSE">license</a>. </i>
*
*
* @section intro Introduction
* The Combinatorial BLAS (CombBLAS) is an extensible distributed-memory parallel graph library offering a small but powerful set of linear
* algebra primitives specifically targeting graph analytics. 
* - The Combinatorial BLAS development influences the <a href="http://graphblas.org">Graph BLAS</a> standardization process.
* - It achieves scalability via its two dimensional distribution and coarse-grained parallelism. 
* - For an illustrative overview, check out <a href="http://eecs.berkeley.edu/~aydin/talks/CombBLAS_Nov11.pdf">these slides</a>. 
* - CombBLAS powers <a href="https://bitbucket.org/azadcse/hipmcl">HipMCL</a>, a highly-scalable parallel implementation of the Markov Cluster Algorithm (MCL).
* - Operations among sparse matrices and vectors use arbitrary user defined semirings. Here is a semiring <a href="http://kdt.sourceforge.net/wiki/index.php/Using_Semirings">primer</a>
* - Check out the <a href="http://eecs.berkeley.edu/~aydin/CombBLAS_FILES/FAQ-combblas.html">Frequently asked questions about CombBLAS</a>.
*
* <b>Download</b> 
* - Read <a href="http://eecs.berkeley.edu/~aydin/CombBLAS_FILES/release-notes.html">release notes</a>.
* - The latest CMake'd tarball (version 1.6.2, April 2018) <a href="http://eecs.berkeley.edu/~aydin/CombBLAS_FILES/CombBLAS_beta_16_2.tgz"> here</a>. (NERSC users read <a href="http://eecs.berkeley.edu/~aydin/CombBLAS_FILES/NERSC_INSTALL.html">this</a>).
 The previous version (version 1.6.1, Jan 2018) is also available <a href="http://eecs.berkeley.edu/~aydin/CombBLAS_FILES/CombBLAS_beta_16_1.tgz"> here </a> for backwards compatibility and benchmarking.
* - Installation and testing can be done by executing these commands within the CombBLAS directory: 
* 		- <i> mkdir _build </i>
*		- <i> mkdir _install </i>
*		- <i> cd _build </i>
* 		- <i> cmake .. -DCMAKE_INSTALL_PREFIX=../_install</i>
* 		- <i> make </i>
* 		- <i> make install </i>
* 		- <i> ctest -V </i> (you need the testinputs, see below)
* 	- Test inputs are separately downloadable <a href="http://eecs.berkeley.edu/~aydin/CombBLAS_FILES/testdata_combblas1.6.1.tgz"> here</a>. Extract them inside the _build directory you've just created with the command "tar -xzvf testdata_combblas1.6.1.tgz"
* - Alternatively (if cmake fails, or you just don't want to install it), you can just imitate the sample makefiles inside the ReleaseTests and Applications 
* directories. Those sample makefiles have the following format: makefile-<i>machine</i>. (example: makefile-macair)
* - The CMake now automatically compiles for hybrid MPI+OpenMP mode because almost all expensive primitives are now multithreaded. Example makefiles are also multithreaded for many cases. You just need to make sure that your OMP_NUM_THREADS environmental variable is set to the right value for the configuration you are running and you are not oversubscribing or undersubscribing cores. 
* - At this point, you can incorporate CombBLAS into your own code by linking against the contents of the <i>_install/lib</i> directory and including the header <i>_install/include/CombBLAS/CombBLAS.h</i>. If you need an example, <a href="https://github.com/xiaoyeli/superlu_dist">SuperLU_Dist</a> does that 
*
* - While we do not recommend using the code from our development repository, you can certainly issue pull requests <a href="https://bitbucket.org/berkeleylab/combinatorial-blas-2.0">there</a>
* 
* <b>Requirements</b>: You need a recent 
* C++ compiler (gcc version 4.8+, Intel version 15.0+ and compatible), a compliant MPI implementation, and C++11 Standard library (libstdc++ that comes with g++
* has them). The recommended tarball uses the CMake build system, but only to build the documentation and unit-tests, and to automate installation. The chances are that you're not going to use any of our sample applications "as-is", so you can just modify them or imitate their structure to write your own application by just using the header files. There are very few binary libraries to link to, and no configured header files. Like many high-performance C++ libraries, the Combinatorial BLAS is mostly templated.
*
* <b>Documentation</b>:
* This is a beta implementation of the Combinatorial BLAS Library in written in C++ with MPI and OpenMP for parallelism.
* It is purposefully designed for distributed memory platforms though it also runs in uniprocessor and shared-memory (such as multicores) platforms. 
* It contains efficient implementations of novel data structures/algorithms
* as well as reimplementations of some previously known data structures/algorithms for convenience. More details can be found in the accompanying paper [1]. One of
* the distinguishing features of the Combinatorial BLAS is its decoupling of parallel logic from the
* sequential parts of the computation, making it possible to implement new formats and plug them in
* without changing the rest of the library.
*
* For I/O purposes, the implementation supports both a tuples format very similar to the Matrix Market and the Matrix Market format itself. We recommend using the Matrix Market version and associated ParallelReadMM() functions.
* We encourage in-memory generators for faster benchmarking. 
*
* The main data structure is a distributed sparse matrix ( SpParMat <IT,NT,DER> ) which HAS-A sequential sparse matrix ( SpMat <IT,NT> ) that 
* can be implemented in various ways as long as it supports the interface of the base class (currently: SpTuples, SpCCols, SpDCCols).
*
* For example, the standard way to declare a parallel sparse matrix A that uses 32-bit integers for indices, floats for numerical values (nonzeros),
* SpDCCols <int,float> for the underlying sequential matrix operations is: 
* - SpParMat<int, float, SpDCCols<int,float> > A; 
*
*
* Sparse and dense vectors are distributed along all processors. This is very space efficient and provides 
* good load balance for SpMSV (sparse matrix-sparse vector multiplication).
*
* <b> New in version 1.6</b>:
* - In-node multithreading enabled for many expensive operations.
* - Fully parallel text-file reader for vectors (FullyDistSpVec::ParallelReadMM() and FullyDistVec::ParallelReadMM())
* - Fully parallel text-file writer for vectors (FullyDistSpVec::ParallelWrite () and FullyDistVec::ParallelWrite())
* - Reverse Cuthill-McKee (RCM) ordering implementation. Please cite [12] if you use this implementation
* - Novel multithreaded SpGEMM and SpMV (with sparse vectors) algorithms are integrated with the rest of CombBLAS.
* 	- For benchmarking multithreaded SpMV with sparse vectors, go to Applications/SpMSpV-IPDPS2017 directory and use the code there.
* 	- Please cite [13] if you use the new multithreaded SpMV with sparse vectors.
* - Extended CSC support
* - Previously deprecated SpParVec and DenseParVec (that were distributed to diagonal processors only) classes are removed.
* - Lots of more bug fixes
*

* <b> New in version 1.5</b>:
* - Fully parallel matrix market format reader (SpParMat::ParallelReadMM())
* - Complete multithreading support, including SpGEMM (previously it was solely SpMV), enabled by -DTHREADED during compilation
* - Experimental 3D SpGEMM (the ability to switch processor grids from 2D to 3D will have to wait for version 1.6)
*       - Cite [9] if you use this implementation
*       - cd 3DSpGEMM/, make test_mpipspgemm, and call the executable with correct parameters
* - Maximal and Maximum cardinality matching algorithms on bipartite graphs
*       - Cite [10] for maximal cardinality and [11] for maximum cardinality matching
*       - cd MaximumMatching, make bpmm, and call the executable with correct parameters
* - Automated MPI_Op creation from simple C++ function objects (simplifies semiring descriptions and Reduce() functions)
* - FullyDistSpVec::Invert() to map from/to (integer) values to/from indices
* - Many more helper functions
* - Experimental CSC support for low concurrencies
* - Lots of bug fixes
*

* <b> New in version 1.4</b>:
* - Direction optimizing breadth-first search in distributed memory (in collaboration with <a href="http://www.cs.berkeley.edu/~sbeamer/">Scott Beamer</a> and <a href="http://www.cs.berkeley.edu/~sbeamer/gap/">GAP</a>). Please cite [8] if you use this code in your research or benchmarks (DirOptBFS.cpp).
*
*
* The supported operations (a growing list) are:
* - Sparse matrix-matrix multiplication on a semiring SR: PSpGEMM()
* - Elementwise multiplication of sparse matrices (A .* B and A .* not(B) in Matlab): EWiseMult()
* - Unary operations on nonzeros: SpParMat::Apply()
* - Matrix-matrix and matrix-vector scaling (the latter scales each row/column with the same scalar of the vector) 
* - Reductions along row/column: SpParMat::Reduce()
* - Sparse matrix-dense vector multiplication on a semiring, SpMV()
* - Sparse matrix-sparse vector multiplication on a semiring, SpMV()
* - Generalized matrix indexing: SpParMat::operator(const FullyDistVec & ri, const FullyDistVec & ci)
* - Generalized sparse matrix assignment: SpParMat::SpAsgn (const FullyDistVec & ri, const FullyDistVec &ci, SpParMat & B)
* - Numeric type conversion through conversion operators
* - Elementwise operations between sparse and dense matrices: SpParMat::EWiseScale() and operator+=()  
* - BFS specific optimizations inside BFSFriends.h
* 
* All the binary operations can be performed on matrices with different numerical value representations.
* The type-traits mechanism will take care of the automatic type promotion, and automatic MPI data type determination.
*
* Some features it uses:
* - templates (for generic types, and for metaprogramming through "curiously recurring template pattern")
* - operator overloading 
* - compositors (to avoid intermediate copying) 
* - standard library whenever possible
* - Reference counting using shared_ptr for IITO (implemented in terms of) relationships
* - As external code, it utilizes 
*	- sequence heaps of <a href="http://www.mpi-inf.mpg.de/~sanders/programs/"> Peter Sanders </a>.
*	- a modified (memory efficient) version of the Viral Shah's <a href="http://www.allthingshpc.org/Publications/psort/psort.html"> PSort </a>.
*	- a modified version of the R-MAT generator from <a href="http://graph500.org"> Graph 500 reference implementation </a>
* 
* Important Sequential classes:
* - SpTuples		: uses triples format to store matrices, mostly used for input/output and intermediate tasks (such as sorting)
* - SpDCCols		: implements Alg 1B and Alg 2 [2], holds DCSC.
* - SpCCols 		: implements CSC

* Important Parallel classes:
* - SpParMat		: distributed memory MPI implementation 
	\n Each processor locally stores its submatrix (block) as a sequential SpDCCols object
	\n Uses a polyalgorithm for SpGEMM: For most systems this boils down to a BSP like Sparse SUMMA [3] algorithm.
* - FullyDistVec	: dense vector distributed to all processors
* - FullyDistSpVec:	: sparse vector distributed to all processors
*
* 
* <b> Applications </b>  implemented using Combinatorial BLAS:
* - BetwCent.cpp : Betweenness centrality computation on directed, unweighted graphs. Download sample input <a href=" http://eecs.berkeley.edu/~aydin/CombBLAS_FILES/scale17_bc_inp.tar.gz"> here </a>.
* - TopDownBFS.cpp: A conformant implementation of the <a href="http://graph500.org">Graph 500 benchmark</a> using the traditional top-down BFS.
* - DirOptBFS.cpp: A conformant implementation of the <a href="http://graph500.org">Graph 500 benchmark</a> using the faster direction-optimizing BFS.
* - FilteredMIS.cpp: Filtered maximal independent set calculation on ER graphs using Luby's algorithm. 
* - FilteredBFS.cpp: Filtered breadth-first search on a twitter-like data set. 
* - BipartiteMatchings/BPMaximalMatching.cpp: Maximal matching algorithms on bipartite graphs [10]
* - BipartiteMatchings/BPMaximumMatching.cpp: Maximum matching algorithm on bipartite graphs [11]
* - Ordering/RCM.cpp: Reverse Cuthill-McKee ordering on distributed memory [12]
*
* <b> Performance </b> results of the first two applications can be found in the design paper [1]; Graph 500 results are in a recent BFS paper [4]. The most
recent sparse matrix indexing, assignment, and multiplication results can be found in [5]. Performance of filtered graph algorithms (BFS and MIS) are reported in [7].
 Performance of the 3D SpGEMM algorithm can be found in [9]
*
* A subset of test programs demonstrating how to use the library (under ReleaseTests):
* - TransposeTest.cpp : File I/O and parallel transpose tests
* - MultTiming.cpp : Parallel SpGEMM tests
* - IndexingTest.cpp: Various sparse matrix indexing usages
* - SpAsgnTiming.cpp: Sparse matrix assignment usage and timing. 
* - FindSparse.cpp : Parallel find/sparse routines akin to Matlab's.
* - GalerkinNew.cpp : Graph contraction or restriction operator (used in Algebraic Multigrid). 
*
* <b> Citation: </b> Please cite the design paper [1] if you end up using the Combinatorial BLAS in your research.
*
* - [1] Aydın Buluç and John R. Gilbert, <i> The Combinatorial BLAS: Design, implementation, and applications </i>. International Journal of High Performance Computing Applications (IJHPCA), 2011. <a href="http://gauss.cs.ucsb.edu/~aydin/combblas-r2.pdf"> Preprint </a>, <a href="http://hpc.sagepub.com/content/early/2011/05/11/1094342011403516.abstract">Link</a>
* - [2] Aydın Buluç and John R. Gilbert, <i> On the Representation and Multiplication of Hypersparse Matrices </i>. The 22nd IEEE International Parallel and Distributed Processing Symposium (IPDPS 2008), Miami, FL, April 14-18, 2008
* - [3] Aydın Buluç and John R. Gilbert, <i> Challenges and Advances in Parallel Sparse Matrix-Matrix Multiplication </i>. The 37th International Conference on Parallel Processing (ICPP 2008), Portland, Oregon, USA, 2008
* - [4] Aydın Buluç and Kamesh Madduri, <i> Parallel Breadth-First Search on Distributed-Memory Systems </i>. Supercomputing (SC'11), Seattle, USA. <a href="http://arxiv.org/abs/1104.4518">Extended preprint</a> <a href="http://gauss.cs.ucsb.edu/~aydin/sc11_bfs.pdf"> PDF </a>
* - [5] Aydın Buluç and John R. Gilbert. <i> Parallel Sparse Matrix-Matrix Multiplication and Indexing: Implementation and Experiments </i>. SIAM Journal of Scientific Computing, 2012. <a href="http://arxiv.org/abs/1109.3739"> Preprint </a>
 <a href="http://gauss.cs.ucsb.edu/~aydin/spgemm_sisc12.pdf"> PDF </a>
* - [6] Aydın Buluç. <i> Linear Algebraic Primitives for Computation on Large Graphs </i>. PhD thesis, University of California, Santa Barbara, 2010. <a href="http://gauss.cs.ucsb.edu/~aydin/Buluc_Dissertation.pdf"> PDF </a>
* - [7] Aydın Buluç, Erika Duriakova, Armando Fox, John Gilbert, Shoaib Kamil, Adam Lugowski, Leonid Oliker, Samuel Williams. <i> High-Productivity and High-Performance Analysis of Filtered Semantic Graphs </i> , International Parallel and Distributed Processing Symposium (IPDPS), 2013. <a href="http://gauss.cs.ucsb.edu/~aydin/ipdps13-kdtsejits.pdf"> PDF </a>
* - [8] Scott Beamer, Aydin Buluç, Krste Asanović, and David Patterson. Distributed memory breadth-first search revisited: Enabling bottom-up search. In Workshop on Multithreaded Architectures and Applications (MTAAP), in conjunction with IPDPS. IEEE Computer Society, 2013.  <a href="http://crd.lbl.gov/assets/pubs_presos/mtaapbottomup2D.pdf"> PDF </a>
* - [9] Ariful Azad, Grey Ballard, Aydin Buluç, James Demmel, Laura Grigori, Oded Schwartz, Sivan Toledo, and Samuel Williams. Exploiting multiple levels of parallelism in sparse matrix-matrix multiplication. SIAM Journal on Scientific Computing (SISC), 38(6):C624--C651, 2016.  <a href="http://gauss.cs.ucsb.edu/~aydin/M104253.pdf"> PDF </a>
* - [10] Ariful Azad and Aydin Buluç. Distributed-memory algorithms for maximal cardinality matching using matrix algebra. In IEEE International Conference on Cluster Computing (CLUSTER), 2015.  <a href="http://gauss.cs.ucsb.edu/~aydin/maximalMatching.pdf"> PDF </a>
* - [11] Ariful Azad and Aydin Buluc. Distributed-memory algorithms for maximum cardinality matching in bipartite graphs. In Proceedings of the IPDPS, 2016.  <a href="http://gauss.cs.ucsb.edu/~aydin/MCM_IPDPS16_Azad.pdf"> PDF </a>
* - [12] Ariful Azad, Mathias Jacquelin, Aydin Buluç, and Esmond G. Ng. The reverse Cuthill-McKee algorithm in distributed-memory. In Proceedings of the IPDPS, 2017. <a href="http://gauss.cs.ucsb.edu/~aydin/RCM-ipdps17.pdf"> PDF </a>
* - [13] Ariful Azad and Aydin Buluç. A work-efficient parallel sparse matrix-sparse vector multiplication algorithm. In Proceedings of the IPDPS, 2017. <a
href="http://gauss.cs.ucsb.edu/~aydin/SpMSpV-ipdps17.pdf"> PDF </a>
*/
