* This is the development repository of Combinatorial BLAS. 

**Copyright** 

Combinatorial BLAS, Copyright (c) 2020, The Regents of the University of California, through Lawrence Berkeley National Laboratory (subject to receipt of any required approvals from the U.S. Dept. of Energy) and University of California, Santa Barbara. All rights reserved.

If you have questions about your rights to use or distribute this software, please contact Berkeley Lab's Innovation & Partnerships Office.

NOTICE. This Software was developed under funding from the U.S. Department of Energy and the U.S. Government consequently retains certain rights. As such, the U.S. Government has been granted for itself and others acting on its behalf a paid-up, nonexclusive, irrevocable, worldwide license in the Software to reproduce, distribute copies to the public, prepare derivative works, and perform publicly and display publicly, and to permit other to do so.

This material is based upon work supported by the National Science Foundation under Grant No. 0709385 and by the Department of Energy, Office of Science, ASCR Contract No. DE-AC02-05CH11231\. Any opinions, findings and conclusions or recommendations expressed in this material are those of the author(s) and do not necessarily reflect the views of the National Science Foundation (NSF) and the Department of Energy (DOE). This software is released under the following [license](https://github.com/PASSIONLab/CombBLAS/blob/master/LICENSE).

**Introduction**

The Combinatorial BLAS (CombBLAS) is an extensible distributed-memory parallel graph library offering a small but powerful set of linear algebra primitives specifically targeting graph analytics. This repo has the code that represents version 2.0 of the software.

*   The Combinatorial BLAS development influences the [Graph BLAS](http://graphblas.org) standardization process.
*   It achieves scalability via its two dimensional distribution and coarse-grained parallelism.
*   For an illustrative overview, check out [these slides](http://eecs.berkeley.edu/~aydin/talks/CombBLAS_Nov11.pdf).
*   CombBLAS powers [HipMCL](https://bitbucket.org/azadcse/hipmcl), a highly-scalable parallel implementation of the Markov Cluster Algorithm (MCL).
*   Operations among sparse matrices and vectors use arbitrary user defined semirings. Here is a semiring [primer](http://kdt.sourceforge.net/wiki/index.php/Using_Semirings)
*   Check out the [Frequently asked questions about CombBLAS](FAQ.md).

**Download**

*   Just run git clone https://github.com/PASSIONLab/CombBLAS.git for the latest code
*   NERSC users read [this](http://eecs.berkeley.edu/~aydin/CombBLAS_FILES/NERSC_INSTALL.html)
*   The old CMake'd tarball (version 1.6.2, April 2018) [here](http://eecs.berkeley.edu/~aydin/CombBLAS_FILES/CombBLAS_beta_16_2.tgz). An even earlier version (version 1.6.1, Jan 2018) is also available [here](http://eecs.berkeley.edu/~aydin/CombBLAS_FILES/CombBLAS_beta_16_1.tgz) for backwards compatibility and benchmarking. Read [release notes](http://eecs.berkeley.edu/~aydin/CombBLAS_FILES/release-notes.html).
*   Installation and testing can be done by executing these commands within the CombBLAS directory:
    1.   mkdir _build
    2.   mkdir _install
    3.   cd _build
    4.   cmake .. -DCMAKE_INSTALL_PREFIX=../_install
    5.   make
    6.   make install
    7.   ctest -V (you need the testinputs, see below)

If running on a Mac, we recommend using gcc compilers instead of clang (which has issues with OpenMP). For that, all you need to do is to replace step (4) above with

*    cmake .. -DCMAKE_INSTALL_PREFIX=../_install  -DCMAKE_C_COMPILER=gcc -DCMAKE_CXX_COMPILER=g++

Test inputs are separately downloadable [here](http://eecs.berkeley.edu/~aydin/CombBLAS_FILES/testdata_combblas1.6.1.tgz). Extract them inside the _build directory you've just created with the command "tar -xzvf testdata_combblas1.6.1.tgz"

*   Alternatively (if cmake fails, or you just don't want to install it), you can just imitate the sample makefiles inside the ReleaseTests and Applications directories. Those sample makefiles have the following format: makefile-_machine_. (example: makefile-macair)
*   The CMake now automatically compiles for hybrid MPI+OpenMP mode because almost all expensive primitives are now multithreaded. Example makefiles are also multithreaded for many cases. You just need to make sure that your OMP_NUM_THREADS environmental variable is set to the right value for the configuration you are running and you are not oversubscribing or undersubscribing cores.
*   At this point, you can incorporate CombBLAS into your own code by linking against the contents of the __install/lib_ directory and including the header __install/include/CombBLAS/CombBLAS.h_. If you need an example, [SuperLU_Dist](https://github.com/xiaoyeli/superlu_dist) does that

**Requirements**: You need a recent C++ compiler (gcc version 4.8+, Intel version 15.0+ and compatible), a compliant MPI implementation, and C++11 Standard library (libstdc++ that comes with g++ has them). The recommended tarball uses the CMake build system, but only to build the documentation and unit-tests, and to automate installation. The chances are that you're not going to use any of our sample applications "as-is", so you can just modify them or imitate their structure to write your own application by just using the header files. There are very few binary libraries to link to, and no configured header files. Like many high-performance C++ libraries, the Combinatorial BLAS is mostly templated.

**Documentation**: This is a beta implementation of the Combinatorial BLAS Library in written in C++ with MPI and OpenMP for parallelism. It is purposefully designed for distributed memory platforms though it also runs in uniprocessor and shared-memory (such as multicores) platforms. It contains efficient implementations of novel data structures/algorithms as well as reimplementations of some previously known data structures/algorithms for convenience. More details can be found in the accompanying paper [1]. One of the distinguishing features of the Combinatorial BLAS is its decoupling of parallel logic from the sequential parts of the computation, making it possible to implement new formats and plug them in without changing the rest of the library.

For I/O purposes, the implementation supports both a tuples format very similar to the Matrix Market and the Matrix Market format itself. We recommend using the Matrix Market version and associated ParallelReadMM() functions. We encourage in-memory generators for faster benchmarking.

The main data structure is a distributed sparse matrix ( SpParMat <IT,NT,DER> ) which HAS-A sequential sparse matrix ( SpMat <IT,NT> ) that can be implemented in various ways as long as it supports the interface of the base class (currently: SpTuples, SpCCols, SpDCCols).

For example, the standard way to declare a parallel sparse matrix A that uses 32-bit integers for indices, floats for numerical values (nonzeros), SpDCCols <int,float> for the underlying sequential matrix operations is:

*   SpParMat<int, float, SpDCCols<int,float> > A;

Sparse and dense vectors are distributed along all processors. This is very space efficient and provides good load balance for SpMSV (sparse matrix-sparse vector multiplication).

**New since version 1.6**:

*   Connected components in distributed memory, found in Applications/CC.h [15,16], compile with "make cc" in that folder. Usage self explanatory (just try ./cc without any parameters to get usage)
*   Incorporation of much faster shared-memory hash SpGEMM implementation [17] from [Yusuke Nagasaka](https://bitbucket.org/YusukeNagasaka/mtspgemmlib/src/master/)
*   Initial CUDA support (for HipMCL initially [18]) for sparse matrix-matrix multiplication
*   3D process grid support for reducing communication in sparse matrix-matrix multiplication [19]

**New in version 1.6**:

*   In-node multithreading enabled for many expensive operations.
*   Fully parallel text-file reader for vectors (FullyDistSpVec::ParallelReadMM() and FullyDistVec::ParallelReadMM())
*   Fully parallel text-file writer for vectors (FullyDistSpVec::ParallelWrite () and FullyDistVec::ParallelWrite())
*   Reverse Cuthill-McKee (RCM) ordering implementation. Please cite [13] if you use this implementation
*   Novel multithreaded SpGEMM and SpMV (with sparse vectors) algorithms are integrated with the rest of CombBLAS.
    *   For benchmarking multithreaded SpMV with sparse vectors, go to Applications/SpMSpV-IPDPS2017 directory and use the code there.
    *   Please cite [14] if you use the new multithreaded SpMV with sparse vectors.
*   Extended CSC support
*   Previously deprecated SpParVec and DenseParVec (that were distributed to diagonal processors only) classes are removed.
*   Lots of more bug fixes

**New in version 1.5**:

*   Fully parallel matrix market format reader (SpParMat::ParallelReadMM())
*   Complete multithreading support, including SpGEMM (previously it was solely SpMV), enabled by -DTHREADED during compilation
*   Experimental 3D SpGEMM (the ability to switch processor grids from 2D to 3D will have to wait for version 1.6)
    *   Cite [10] if you use this implementation
    *   cd 3DSpGEMM/, make test_mpipspgemm, and call the executable with correct parameters
*   Maximal and Maximum cardinality matching algorithms on bipartite graphs
    *   Cite [11] for maximal cardinality and [12] for maximum cardinality matching
    *   cd MaximumMatching, make bpmm, and call the executable with correct parameters
*   Automated MPI_Op creation from simple C++ function objects (simplifies semiring descriptions and Reduce() functions)
*   FullyDistSpVec::Invert() to map from/to (integer) values to/from indices
*   Many more helper functions
*   Experimental CSC support for low concurrencies
*   Lots of bug fixes

**New in version 1.4**:

*   Direction optimizing breadth-first search in distributed memory (in collaboration with [Scott Beamer](http://www.cs.berkeley.edu/~sbeamer/) and [GAP](http://www.cs.berkeley.edu/~sbeamer/gap/)). Please cite [8] if you use this code in your research or benchmarks ([DirOptBFS.cpp](_dir_opt_b_f_s_8cpp.html)).

The supported operations (a growing list) are:

*   Sparse matrix-matrix multiplication on a semiring SR: [PSpGEMM()](namespacecombblas.html#a4683888892943d76bd707bf5e4b11f15)
*   Elementwise multiplication of sparse matrices (A .* B and A .* not(B) in Matlab): [EWiseMult()](namespacecombblas.html#a1fca28136b736b66fea4f09e01b199c5)
*   Unary operations on nonzeros: SpParMat::Apply()
*   Matrix-matrix and matrix-vector scaling (the latter scales each row/column with the same scalar of the vector)
*   Reductions along row/column: SpParMat::Reduce()
*   Sparse matrix-dense vector multiplication on a semiring, [SpMV()](papi__combblas__globals_8h.html#a40d63e35a8bec1195af2124e1dc6b61fac41d9bd18e43d4d7eb692f86029f2ce1)
*   Sparse matrix-sparse vector multiplication on a semiring, [SpMV()](papi__combblas__globals_8h.html#a40d63e35a8bec1195af2124e1dc6b61fac41d9bd18e43d4d7eb692f86029f2ce1)
*   Generalized matrix indexing: SpParMat::operator(const FullyDistVec & ri, const FullyDistVec & ci)
*   Generalized sparse matrix assignment: SpParMat::SpAsgn (const FullyDistVec & ri, const FullyDistVec &ci, SpParMat & B)
*   Numeric type conversion through conversion operators
*   Elementwise operations between sparse and dense matrices: SpParMat::EWiseScale() and operator+=()
*   BFS specific optimizations inside [BFSFriends.h](_b_f_s_friends_8h.html)

All the binary operations can be performed on matrices with different numerical value representations. The type-traits mechanism will take care of the automatic type promotion, and automatic MPI data type determination.

Some features it uses:

*   templates (for generic types, and for metaprogramming through "curiously recurring template pattern")
*   operator overloading
*   compositors (to avoid intermediate copying)
*   standard library whenever possible
*   Reference counting using shared_ptr for IITO (implemented in terms of) relationships
*   As external code, it utilizes
    *   sequence heaps of [Peter Sanders](http://www.mpi-inf.mpg.de/~sanders/programs/) .
    *   a modified (memory efficient) version of the Viral Shah's [PSort](http://www.allthingshpc.org/Publications/psort/psort.html) .
    *   a modified version of the R-MAT generator from [Graph 500 reference implementation](http://graph500.org)

Important Sequential classes:

*   SpTuples : uses triples format to store matrices, mostly used for input/output and intermediate tasks (such as sorting)
*   SpDCCols : implements Alg 1B and Alg 2 [3], holds DCSC.
*   SpCCols : implements CSC

Important Parallel classes:

*   SpParMat : distributed memory MPI implementation  
    Each processor locally stores its submatrix (block) as a sequential SpDCCols object  
    Uses a polyalgorithm for SpGEMM: For most systems this boils down to a BSP like Sparse SUMMA [4] algorithm.
*   SpParMat3D : sparse matrix distributed in 3D process grid
*   FullyDistVec : dense vector distributed to all processors
*   FullyDistSpVec: : sparse vector distributed to all processors

**Applications** implemented using Combinatorial BLAS:

*   [BetwCent.cpp](_betw_cent_8cpp.html) : Betweenness centrality computation on directed, unweighted graphs. Download sample input [here]( http://eecs.berkeley.edu/~aydin/CombBLAS_FILES/scale17_bc_inp.tar.gz) .
*   [TopDownBFS.cpp](_top_down_b_f_s_8cpp.html): A conformant implementation of the [Graph 500 benchmark](http://graph500.org) using the traditional top-down BFS.
*   [DirOptBFS.cpp](_dir_opt_b_f_s_8cpp.html): A conformant implementation of the [Graph 500 benchmark](http://graph500.org) using the faster direction-optimizing BFS.
*   [FilteredMIS.cpp](_filtered_m_i_s_8cpp.html): Filtered maximal independent set calculation on ER graphs using Luby's algorithm.
*   [FilteredBFS.cpp](_filtered_b_f_s_8cpp.html): Filtered breadth-first search on a twitter-like data set.
*   [BipartiteMatchings/BPMaximalMatching.cpp](_b_p_maximal_matching_8cpp.html): Maximal matching algorithms on bipartite graphs [11]
*   [BipartiteMatchings/BPMaximumMatching.cpp](_b_p_maximum_matching_8cpp.html): Maximum matching algorithm on bipartite graphs [12]
*   [Ordering/RCM.cpp](_r_c_m_8cpp.html): Reverse Cuthill-McKee ordering on distributed memory [13]
*   [CC.cpp](https://bitbucket.org/berkeleylab/combinatorial-blas-2.0/src/master/CombBLAS/Applications/CC.cpp): Linear-algebraic connected components [15, 16]
*   [MCL3D.cpp](https://github.com/PASSIONLab/CombBLAS/blob/master/Applications/MCL3D.cpp): HipMCL using 3D process grid

**Performance** results of the first two applications can be found in the design paper [1]; Graph 500 results are in a recent BFS paper [5]. The sparse matrix indexing, assignment, and multiplication results using 2D algorithms can be found in [6]. Performance of filtered graph algorithms (BFS and MIS) are reported in [8]. Performance of the 3D SpGEMM algorithm can be found in [10]

A subset of test programs demonstrating how to use the library (under ReleaseTests):

*   [TransposeTest.cpp](_transpose_test_8cpp.html) : File I/O and parallel transpose tests
*   [MultTiming.cpp](_mult_timing_8cpp.html) : Parallel SpGEMM tests
*   [IndexingTest.cpp](_indexing_test_8cpp.html): Various sparse matrix indexing usages
*   [ParIOTest.cpp](https://bitbucket.org/berkeleylab/combinatorial-blas-2.0/src/master/CombBLAS/ReleaseTests/ParIOTest.cpp) Parallel reading of arbitrary labeled tuples with SpParMat::ReadGeneralizedTuples()
*   [ReadWriteMtx.cpp](https://bitbucket.org/berkeleylab/combinatorial-blas-2.0/src/master/CombBLAS/ReleaseTests/ReadWriteMtx.cpp): Parallel matrix-market I/O
*   [VectorIO.cpp](https://bitbucket.org/berkeleylab/combinatorial-blas-2.0/src/master/CombBLAS/ReleaseTests/VectorIO.cpp): Parallel Vector I/O
*   [GenWriteMatrix.cpp](https://bitbucket.org/berkeleylab/combinatorial-blas-2.0/src/master/CombBLAS/ReleaseTests/GenWriteMatrix.cpp): Parallel generating and writing of Kronecker graphs
*   [SpAsgnTiming.cpp](_sp_asgn_timing_8cpp.html): Sparse matrix assignment usage and timing.
*   [FindSparse.cpp](_find_sparse_8cpp.html) : Parallel find/sparse routines akin to Matlab's.
*   [GalerkinNew.cpp](_galerkin_new_8cpp.html) : Graph contraction or restriction operator (used in Algebraic Multigrid).
*   [Application/SpGEMM3D.cpp](https://github.com/PASSIONLab/CombBLAS/blob/master/Applications/SpGEMM3D.cpp) : Using sparse matrix-matrix multiplication using 3D process grid.

**Citation:** Please cite the current CombBLAS 2.0 paper [2] if you end up using the Combinatorial BLAS in your research in 2021 or later. If you are simply referencing the design principles and the concepts behind CombBLAS, then you can either cite [1] or [2], depending on the concept. 

*   [1] Aydin Buluc and John R. Gilbert, _The Combinatorial BLAS: Design, implementation, and applications_ . International Journal of High Performance Computing Applications (IJHPCA), 2011\. [Preprint](https://people.eecs.berkeley.edu/~aydin/combblas-r2.pdf) , [Link](http://hpc.sagepub.com/content/early/2011/05/11/1094342011403516.abstract)
*   [2] Ariful Azad, Oguz Selvitopi, Md Taufique Hussain, John R. Gilbert, Aydin Buluç. _Combinatorial BLAS 2.0: Scaling combinatorial algorithms on distributed-memory systems_. IEEE Transactions on Parallel and Distributed Systems (TPDS), Early accesss, 2021. [Preprint](https://arxiv.org/pdf/2106.14402.pdf) , [Link](https://ieeexplore.ieee.org/abstract/document/9470983)
*   [3] Aydin Buluc and John R. Gilbert, _On the Representation and Multiplication of Hypersparse Matrices_ . The 22nd IEEE International Parallel and Distributed Processing Symposium (IPDPS 2008), Miami, FL, April 14-18, 2008
*   [4] Aydin Buluc and John R. Gilbert, _Challenges and Advances in Parallel Sparse Matrix-Matrix Multiplication_ . The 37th International Conference on Parallel Processing (ICPP 2008), Portland, Oregon, USA, 2008
*   [5] Aydin Buluc and Kamesh Madduri, _Parallel Breadth-First Search on Distributed-Memory Systems_ . Supercomputing (SC'11), Seattle, USA. [Extended preprint](http://arxiv.org/abs/1104.4518) [PDF](https://people.eecs.berkeley.edu/~aydin/sc11_bfs.pdf)
*   [6] Aydin Buluc and John R. Gilbert. _Parallel Sparse Matrix-Matrix Multiplication and Indexing: Implementation and Experiments_ . SIAM Journal of Scientific Computing, 2012\. [Preprint](http://arxiv.org/abs/1109.3739) [PDF](https://people.eecs.berkeley.edu/~aydin/spgemm_sisc12.pdf)
*   [7] Aydin Buluc. _Linear Algebraic Primitives for Computation on Large Graphs_ . PhD thesis, University of California, Santa Barbara, 2010\. [PDF](https://people.eecs.berkeley.edu/~aydin/Buluc_Dissertation.pdf)
*   [8] Aydin Buluc, Erika Duriakova, Armando Fox, John Gilbert, Shoaib Kamil, Adam Lugowski, Leonid Oliker, Samuel Williams. _High-Productivity and High-Performance Analysis of Filtered Semantic Graphs_ , International Parallel and Distributed Processing Symposium (IPDPS), 2013\. [PDF](https://people.eecs.berkeley.edu/~aydin/ipdps13-kdtsejits.pdf)
*   [9] Scott Beamer, Aydin Buluc, Krste Asanovic, and David Patterson. Distributed memory breadth-first search revisited: Enabling bottom-up search. In Workshop on Multithreaded Architectures and Applications (MTAAP), in conjunction with IPDPS. IEEE Computer Society, 2013\. [PDF](http://crd.lbl.gov/assets/pubs_presos/mtaapbottomup2D.pdf)
*   [10] Ariful Azad, Grey Ballard, Aydin Buluc, James Demmel, Laura Grigori, Oded Schwartz, Sivan Toledo, and Samuel Williams. Exploiting multiple levels of parallelism in sparse matrix-matrix multiplication. SIAM Journal on Scientific Computing (SISC), 38(6):C624-C651, 2016\. [PDF](https://people.eecs.berkeley.edu/~aydin/M104253.pdf)
*   [11] Ariful Azad and Aydin Buluc. Distributed-memory algorithms for maximal cardinality matching using matrix algebra. In IEEE International Conference on Cluster Computing (CLUSTER), 2015\. [PDF](https://people.eecs.berkeley.edu/~aydin/maximalMatching.pdf)
*   [12] Ariful Azad and Aydin Buluc. Distributed-memory algorithms for maximum cardinality matching in bipartite graphs. In Proceedings of the IPDPS, 2016\. [PDF](https://people.eecs.berkeley.edu/~aydin/MCM_IPDPS16_Azad.pdf)
*   [13] Ariful Azad, Mathias Jacquelin, Aydin Buluc, and Esmond G. Ng. The reverse Cuthill-McKee algorithm in distributed-memory. In Proceedings of the IPDPS, 2017\. [PDF](https://people.eecs.berkeley.edu/~aydin/RCM-ipdps17.pdf)
*   [14] Ariful Azad and Aydin Buluc. A work-efficient parallel sparse matrix-sparse vector multiplication algorithm. In Proceedings of the IPDPS, 2017\. [PDF](https://people.eecs.berkeley.edu/~aydin/SpMSpV-ipdps17.pdf)
*   [15] Yongzhe Zhang, Ariful Azad, and Aydin Buluc. "Parallel algorithms for finding connected components using linear algebra." Journal of Parallel and Distributed Computing (2020). [PDF](https://escholarship.org/content/qt8ms106vm/qt8ms106vm_noSplash_bd6caa99d078099df438bfe7c3854e2b.pdf)
*   [16] Ariful Azad and Aydin Buluc. LACC: a linear-algebraic algorithm for finding connected components in distributed memory. In Proceedings of the IPDPS, 2019\. [PDF](https://people.eecs.berkeley.edu/~aydin/LACC.pdf)
*   [17] Yusuke Nagasaka, Satoshi Matsuoka, Ariful Azad, and Aydin Buluc. "Performance optimization, modeling and analysis of sparse matrix-matrix products on multi-core and many-core processors." Parallel Computing 90 (2019): 102545 \. [PDF](https://people.eecs.berkeley.edu/~aydin/spgemm_parco2019.pdf)
*   [18] Oguz Selvitopi, Md Taufique Hussain, Ariful Azad, and Aydin Buluc. Optimizing high performance Markov clustering for pre-exascale architectures. In Proceedings of the IPDPS, 2020 \. [PDF](https://people.eecs.berkeley.edu/~aydin/HipMCL_PreExascale-IPDPS20.pdf)
*   [19] Md Taufique Hussain, Oguz Selvitopi, Aydin Buluc, Ariful Azad. Communication-Avoiding and Memory-Constrained Sparse Matrix-Matrix Multiplication at Extreme Scale. \. [PDF](https://arxiv.org/pdf/2010.08526.pdf)


