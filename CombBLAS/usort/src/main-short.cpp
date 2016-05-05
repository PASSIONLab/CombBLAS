#include <mpi.h>

#include <string>
#include <iomanip>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <vector>
#include <omp.h>
#include <sstream>


#ifdef _PROFILE_SORT
#include "sort_profiler.h"
#endif

#include <binUtils.h>
#include <ompUtils.h>
#include <parUtils.h>



int main(int argc, char **argv){

  int num_threads = 1;
  omp_set_num_threads(num_threads);

  // Initialize MPI
    MPI_Init(&argc, &argv);
  int myrank;
  MPI_Comm_rank(MPI_COMM_WORLD, &myrank);





    std::vector<int> in;
    if(myrank==0)
    {
        in.resize(2);
        in[0] = 1;
        in[1] = 2;
    }
 


  double t0 = omp_get_wtime();
  par::sampleSort(in, MPI_COMM_WORLD);
  double t1 = omp_get_wtime();

  std::cout << myrank << ": all done in " << t1-t0 << std::endl;
  
  
  MPI_Finalize();
  return 0;
 
 }


