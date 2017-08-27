
********************************************************************************
*::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::*
*::'#::::'#::'######::::'#####::::'#####:::::::::::::::::::::::::::::::::::::::*
*:: #:::: #:: #:::: #::'#.... #::'#.... #::'#####:::::'##::::'#####:::'#:::'#::*
*:: #:::: #:: #:::: #:: #::::.::: #:::::::: #... #:::'#. #::: #... #:: #::: #::*
*:: #######:: ######::: #:::::::: #:'####:: #::: #::'#::. #:: #::: #:: ######::*
*:: #:::: #:: #:::::::: #:::::::: #:... #:: #####::: ######:: ##### :: #... #::*
*:: #:::: #:: #:::::::: #:::: #:: #:::: #:: #.. #::: #::: #:: #...:::: #::: #::*
*:: #:::: #:: #::::::::. #####:::. #####::: #::. #:: #::: #:: #::::::: #::: #::*
*::.:::::.:::.::::::::::.....:::::.....::::.::::.:::.::::.:::.::::::::.::::.:::*
********************************************************************************

       HPCGraph: Graph Computations on High Performance Computing Systems      
                    Copyright (2016) Sandia Corporation

Questions?  Contact:  George M. Slota    (gmslota@sandia.gov)
                      Siva Rajamanickam  (srajama@sandia.gov)
                      Kamesh Madduri     (madduri@cse.psu.edu)

If used, please cite:

[1] George M. Slota, Sivasankaran Rajamanickam, and Kamesh Madduri , A Case Study of Complex Graph Analysis in Distributed Memory: Implementation and Optimization, in the Proceedings of the 30th IEEE International Parallel and Distributed Processing Symposium (IPDPS16).

********************************************************************************
To make:

1.) Set CXX in Makefile to your MPI c++ compiler, adjust CXXFLAGS if necessary
-OpenMP 3.0 support for your compiler is required

2.) $ make
-This will make the 'hpcgraph' executable


********************************************************************************
To run:

$ mpirun -n [#] ./hpcgraph [graphfile] [options]

-[graphfile] is of binary directed edge list format with 32 or 64 bit unsigned integers as vertex labels; vertex labels are assumed to begin at 0. To switch between 32 and 64 bit, alter load_graph_edges_32() to load_graph_edges_64() in main.cpp. 

A binary file containing a list of unsigned 32 or 64 bit integers like this:
v0 v1 v1 v2 v2 v3 v3 v4

would indicate a graph with the following directed edges:
v0 -> v1
v1 -> v2
v2 -> v3
v3 -> v0

Options are:
-a
  Run all analytics
-w
  Run weakly connected components
-s
  Run strongly connected components
-l
  Run label propagation
-p
  Run PageRank
-c
  Run harmonic centrality
-k
  Run k-core algorithm [set -t to limit iterations for approximate version]
-o [outfile]
  Adjust output file (will append '.algorithm' to output file)
-i [comma separated input vertex id list]
  Vertex ids to analyze for harmonic centrality [default: none]
-p [part file]
  Partition file to use for mapping vertices to tasks
-t [#]
  Adjust iteration count [default: 20]
-f
  Run verification routines
-v
  set verbose output
-d 
  set debug output


-Iteration counts are used in label propagation, PageRank, and approximate k-core. It is suggested to set -t to some high bound for k-core as a fail-safe, as the iterative distributed Monstresor et al. algorithm might run for hundreds to thousands of iterations for highly skewed graphs.

-By default, only the total execution time is output when -o output option is not used. Written output for most algorithms will be 'number of vertices' lines with one value on each line corresponding to each vertex's mapped value. For harmonic centrality, output will be 'vertex, value' pairs based on what was selected for -i

-For more verbose timings and output, use the -v and -d flags


********************************************************************************
Examples:

Run PageRank for default 20 iterations on web.graph
# mpirun -n 16 ./hpcgraph web.graph -r


Run Label Propagation for 30 iterations on web.graph
# mpirun -n 16 ./hpcgraph web.graph -l -t 30


Run K-core for 30 iterations on web.graph [if web.graph requires less than 30 iterations to compute k-cores, the algorithm will become the exact version]
# mpirun -n 16 ./hpcgraph web.graph -k -t 30


Run Label Propagation for 30 iterations on web.graph with partition file
# mpirun -n 16 ./hpcgraph web.graph -l -t 30 -p web.parts.16


Compute harmonic centrality values of vertices 1, 5, and 16 on web.graph, output to web.harmonic
# mpirun -n 16 ./hpcgraph -c -i 1,5,16 -o web


Run all algorithms, decrease iteration count, use partition file, write out results
# mpirun -n 16 ./hpcgraph -a -i 1,5,16 -t 10 -p web.parts.16 -o web


Run all algorithms, decrease iteration count, use partition file, write out results, have verbose and debug output, run verification routines
# mpirun -n 16 ./hpcgraph -a -i 1,5,16 -t 10 -p web.parts.16 -o web -v -d -f
