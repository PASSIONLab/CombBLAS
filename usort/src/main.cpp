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

#ifdef KWICK
	#if KWAY > 2
		#define SORT_FUNCTION par::HyperQuickSort_kway
	#else
		#ifdef SWAPRANKS
      #define SORT_FUNCTION par::RankSwapSort
    #else
	 	  #define SORT_FUNCTION par::HyperQuickSort
    #endif
	#endif
#else
    #define SORT_FUNCTION par::sampleSort
#endif

// #define __VERIFY__


enum DistribType{
	UNIF_DISTRIB,
	GAUSS_DISTRIB,
};

void printResults(int num_threads, MPI_Comm comm);

void getStats(double val, double *meanV, double *minV, double *maxV, MPI_Comm comm) 
{ 
	int p; 
	double d, din;
	din = val;
  MPI_Comm_size(comm, &p);
	MPI_Reduce(&din, &d, 1, MPI_DOUBLE, MPI_SUM, 0, comm); *meanV = d/p;
	MPI_Reduce(&din, &d, 1, MPI_DOUBLE, MPI_MIN, 0, comm); *minV = d;
	MPI_Reduce(&din, &d, 1, MPI_DOUBLE, MPI_MAX, 0, comm); *maxV = d;
}

DistribType getDistType(char* code) {
	if(!strcmp(code,"GAUSS\0")){
		return GAUSS_DISTRIB;
	}else{
		return UNIF_DISTRIB;
	}
}

long getNumElements(char* code) {
  unsigned int slen = strlen(code);
  char dtype = code[0];
  char tmp[128];
  strncpy(tmp, code+1, slen-3); tmp[slen-3] = '\0';

  long numBytes = atol(tmp);
  switch(code[slen-2]) {
    case 'g':
    case 'G':
      numBytes *= 1024*1024*1024;
      break;
    case 'k':
    case 'K':
      numBytes *= 1024;
      break;
    case 'm':
    case 'M':
      numBytes *= 1024*1024;
      break;
    default:
      std::cout << "unknown code " << code[slen-2] << std::endl;
      return 0;
  };

  switch (dtype) {
    case 'd': // double array
      return numBytes/sizeof(double);
      break;
    case 'f': // float array
      return numBytes/sizeof(float);
      break;
    case 'i': // int array
      return numBytes/sizeof(int);
      break;
    case 'l': // long array
      return numBytes/sizeof(long);
      break;
    default:
			std::cout << "unknown dtype: " << dtype << std::endl;
      return 0;
  };

}

template <class T>
bool verify (std::vector<T>& in_, std::vector<T> &out_, MPI_Comm comm){

  // Find out my identity in the default communicator 
  int myrank, p;
  MPI_Comm_rank(comm, &myrank);
  MPI_Comm_size(comm,&p);

  if (!myrank) std::cout << "Verifying sort" << std::endl;

  std::vector<T> in;
  {
    int N_local=in_.size()*sizeof(T);
    std::vector<int> r_size(p, 0);
    std::vector<int> r_disp(p, 0);
    MPI_Gather(&N_local  , 1, MPI_INT, 
        &r_size[0], 1, MPI_INT, 0, comm);
    omp_par::scan(&r_size[0], &r_disp[0], p);

    if(!myrank) in.resize((r_size[p-1]+r_disp[p-1])/sizeof(T));
    MPI_Gatherv((char*)&in_[0],    N_local,             MPI_BYTE, 
        (char*)&in [0], &r_size[0], &r_disp[0], MPI_BYTE, 0, comm);
  }

  std::vector<T> out;
  {
    int N_local=out_.size()*sizeof(T);
    std::vector<int> r_size(p, 0);
    std::vector<int> r_disp(p, 0);
    MPI_Gather(&N_local  , 1, MPI_INT, 
        &r_size[0], 1, MPI_INT, 0, comm);
    omp_par::scan(&r_size[0], &r_disp[0], p);

    if(!myrank) out.resize((r_size[p-1]+r_disp[p-1])/sizeof(T));
    MPI_Gatherv((char*)&out_[0],    N_local,             MPI_BYTE, 
        (char*)&out [0], &r_size[0], &r_disp[0], MPI_BYTE, 0, comm);
  }

  if(in.size()!=out.size()){
    std::cout<<"Wrong size: in="<<in.size()<<" out="<<out.size()<<'\n';
    return false;
  }
  std::sort(&in[0], &in[in.size()]);

  for(long j=0;j<in.size();j++)
    if(in[j]!=out[j]){
      std::cout<<"Failed at:"<<j<<'\n';
//      std::cout<<"Failed at:"<<j<<"; in="<<in[j]<<" out="<<out[j]<<'\n';
      return false;
    }

  return true;
}

template <class T>
double time_sort(size_t N, MPI_Comm comm, DistribType dist_type){
  int myrank, p;

  MPI_Comm_rank(comm, &myrank);
  MPI_Comm_size(comm,&p);
  int omp_p=omp_get_max_threads();

  std::vector<IndexHolder<T>> in1(N);
    
  // Generate random data
  std::vector<T> in(N);
	if(dist_type==UNIF_DISTRIB){
    #pragma omp parallel for
    for(int j=0;j<omp_p;j++){
      unsigned int seed=j*p+myrank;
      size_t start=(j*N)/omp_p;
      size_t end=((j+1)*N)/omp_p;
      for(unsigned int i=start;i<end;i++){ 
        in[i]=rand_r(&seed);
      }
    }
	} else if(dist_type==GAUSS_DISTRIB){
    double e=2.7182818284590452;
    double log_e=log(e);

    unsigned int seed1=p+myrank;
    long mn = rand_r(&seed1);

    #pragma omp parallel for
    for(int j=0;j<omp_p;j++){
      unsigned int seed=j*p+myrank;
      size_t start=(j*N)/omp_p;
      size_t end=((j+1)*N)/omp_p;
      for(unsigned int i=start;i<end;i++){ 
        in[i]= mn + sqrt(-2*log(rand_r(&seed)*1.0/RAND_MAX)/log_e)
              * cos(rand_r(&seed)*2*M_PI/RAND_MAX)*RAND_MAX*0.1;
      }
    }
	}
 
 std::vector<T> in_cpy=in;
  std::vector<T> out;

  // in=in_cpy;
  SORT_FUNCTION<T>(in_cpy, comm);
#ifdef __VERIFY__
  verify(in,in_cpy,comm);
#endif

#ifdef _PROFILE_SORT
	total_sort.clear();
	
	seq_sort.clear();
	sort_partitionw.clear();
	
	sample_get_splitters.clear();
	sample_sort_splitters.clear();
	sample_prepare_scatter.clear();
	sample_do_all2all.clear();
	
	hyper_compute_splitters.clear();
	hyper_communicate.clear();
	hyper_merge.clear();
#endif


  //Sort
  MPI_Barrier(comm);
  double wtime=-omp_get_wtime();
  // SORT_FUNCTION<T>(in, out, comm);
  SORT_FUNCTION<T>(in, comm);
  MPI_Barrier(comm);
  wtime+=omp_get_wtime();

  return wtime;
}


/*
template <class T>
double time_sort(size_t N, MPI_Comm comm, DistribType dist_type){
    int myrank, p;
    
    MPI_Comm_rank(comm, &myrank);
    MPI_Comm_size(comm,&p);
    int omp_p=omp_get_max_threads();
    
    // Generate random data
    std::vector<T> in(N);
    if(dist_type==UNIF_DISTRIB){
#pragma omp parallel for
        for(int j=0;j<omp_p;j++){
            unsigned int seed=j*p+myrank;
            size_t start=(j*N)/omp_p;
            size_t end=((j+1)*N)/omp_p;
            for(unsigned int i=start;i<end;i++){
                in[i]=rand_r(&seed);
            }
        }
    } else if(dist_type==GAUSS_DISTRIB){
        double e=2.7182818284590452;
        double log_e=log(e);
        
        unsigned int seed1=p+myrank;
        long mn = rand_r(&seed1);
        
#pragma omp parallel for
        for(int j=0;j<omp_p;j++){
            unsigned int seed=j*p+myrank;
            size_t start=(j*N)/omp_p;
            size_t end=((j+1)*N)/omp_p;
            for(unsigned int i=start;i<end;i++){
                in[i]= mn + sqrt(-2*log(rand_r(&seed)*1.0/RAND_MAX)/log_e)
                * cos(rand_r(&seed)*2*M_PI/RAND_MAX)*RAND_MAX*0.1;
            }
        }
    }
    
    std::vector<T> in_cpy=in;
    std::vector<T> out;
    
    // in=in_cpy;
    SORT_FUNCTION<T>(in_cpy, comm);
#ifdef __VERIFY__
    verify(in,in_cpy,comm);
#endif
    
#ifdef _PROFILE_SORT
    total_sort.clear();
    
    seq_sort.clear();
    sort_partitionw.clear();
    
    sample_get_splitters.clear();
    sample_sort_splitters.clear();
    sample_prepare_scatter.clear();
    sample_do_all2all.clear();
    
    hyper_compute_splitters.clear();
    hyper_communicate.clear();
    hyper_merge.clear();
#endif
    
    
    //Sort
    MPI_Barrier(comm);
    double wtime=-omp_get_wtime();
    // SORT_FUNCTION<T>(in, out, comm);
    SORT_FUNCTION<T>(in, comm);
    MPI_Barrier(comm);
    wtime+=omp_get_wtime();
    
    return wtime;
}*/

int main(int argc, char **argv){
  
  if (argc < 4) {
    std::cerr << "Usage: " << argv[0] << " numThreads typeSize typeDistrib" << std::endl;
    std::cerr << "\t\t typeSize is a character for type of data follwed by data size per node." << std::endl;
		std::cerr << "\t\t typeSize can be d-double, f-float, i-int, l-long." << std::endl;
    std::cerr << "\t\t Examples:" << std::endl;
    std::cerr << "\t\t i1GB : integer  array of size 1GB" << std::endl;
    std::cerr << "\t\t l1GB : long     array of size 1GB" << std::endl;
		std::cerr << "\t\t typeDistrib can be UNIF, GAUSS" << std::endl;
    return 1;  
  }

  std::cout<<setiosflags(std::ios::fixed)<<std::setprecision(4)<<std::setiosflags(std::ios::right);

  //Set number of OpenMP threads to use.
  int num_threads = atoi(argv[1]);
	omp_set_num_threads(num_threads);

  // Initialize MPI
  MPI_Init(&argc, &argv);

  // Find out my identity in the default communicator 
  int myrank;
  MPI_Comm_rank(MPI_COMM_WORLD, &myrank);

  // Find out number of processes
  int p;
  MPI_Comm_size(MPI_COMM_WORLD, &p);


  //!----------------------------------
  /*
  // 1. read in file corresponding to the node % 100;
  // 2. select splitters ... 5 - 20 - print ...
  // 3. bucket and write ... with num_spliters = 20;



  if (!myrank) std::cout << myrank << ": reading file" << std::endl;
  char fname[256];
  // sprintf(fname, "/scratch/01903/hsundar/from_karl/input/part%d", myrank%100);
  sprintf(fname, "part%d", myrank);
  FILE* fp = fopen(fname, "rb");

  long lSize;
  size_t result;
  fseek (fp , 0 , SEEK_END);
  lSize = ftell (fp);
  rewind (fp);

  // test buckets ... 
  int nl = lSize/100; 
  std::vector<sortRecord> qq(nl), out;
  
  // printf("%d: read size is %ld\n", myrank, lSize);
  / *
  srand(myrank*1713);
  for (int i=0; i<nl; i++) {
    qq[i] = rand();
  }
  * /
  result = fread ( &(*qq.begin()), 1, lSize, fp);
  if (result != lSize) { std::cerr << myrank << ": Reading error" << std::endl; exit (3);}
  fclose (fp);
  if (!myrank) std::cout << myrank << ": finished reading file" << std::endl;

  omp_par::merge_sort(&qq[0], &qq[qq.size()]);

  MPI_Barrier(MPI_COMM_WORLD);
  if (!myrank) std::cout << myrank << ": starting approx Select" << std::endl;
  
  std::vector<std::pair<sortRecord, DendroIntL> > sortBins = par::Sorted_approx_Select_skewed(qq, 9, MPI_COMM_WORLD);

  if (!myrank) std::cout << myrank << ": finished approx Select" << std::endl;
  / *
  if (!myrank) {
    std::cout << "splitters = [" << std::endl;
    for (int i=0; i<9; ++i) std::cout << sortBins[i] << " " << std::endl;;
    std::cout << "]" << std::endl;
  }
  * /
  / *
  double t0 = omp_get_wtime();
  par::bucketDataAndWriteSkewed(qq, sortBins, "/tmp/foo", MPI_COMM_WORLD);
  double t1 = omp_get_wtime();
  
  * /
  double t0 = omp_get_wtime();
  // par::HyperQuickSort(qq, out, MPI_COMM_WORLD);
  par::sampleSort(qq, MPI_COMM_WORLD);
  double t1 = omp_get_wtime();

  std::cout << myrank << ": all done in " << t1-t0 << std::endl;
  
  
  MPI_Finalize();
  return 0;
  //!----------------------------------
  */
  int proc_group=0;
  int min_np=1;
  MPI_Comm comm;

  std::vector<double> tt(10000,0);
  
  int k = 0; // in case size based runs are needed 
  char dtype = argv[2][0];
  long N = getNumElements(argv[2]);
	DistribType dist_type=getDistType(argv[3]);
  if (!N) {
    std::cerr << "illegal typeSize code provided: " << argv[2] << std::endl;
    return 2;
  }
 
  std::string num;
  std::stringstream mystream;
  mystream << N*p;
  num = mystream.str();
  int insertPosition = num.length() - 3;
  while (insertPosition > 0) {
    num.insert(insertPosition, ",");
    insertPosition-=3;
  }
  if (!myrank)
    std::cout << "sorting array of size " << num << " keys of type " << dtype << std::endl;

  // check if arguments are ok ...
    
  { // -- full size run  
    double ttt;
    
    switch(dtype) {
			case 'd':
				ttt = time_sort<double>(N, MPI_COMM_WORLD,dist_type);
				break;
			case 'f':
				ttt = time_sort<float>(N, MPI_COMM_WORLD,dist_type);
				break;	
			case 'i':
				ttt = time_sort<int>(N, MPI_COMM_WORLD,dist_type);
				break;
      case 'l':
        ttt = time_sort<long>(N, MPI_COMM_WORLD,dist_type);
        break;
    };
#ifdef _PROFILE_SORT 			
		if (!myrank) {
			std::cout << "---------------------------------------------------------------------------" << std::endl;
		#ifndef KWICK
			std::cout << "\tSample Sort with " << KWAY << "-way all2all" << "\t\tMean\tMin\tMax" << std::endl;
		#else
		  #ifdef SWAPRANKS
			  std::cout << "\t" << KWAY << "-way SwapRankSort " << "\t\tMean\tMin\tMax" << std::endl;
      #else	
        std::cout << "\t" << KWAY << "-way HyperQuickSort" << "\t\tMean\tMin\tMax" << std::endl;
		  #endif
		#endif
			std::cout << "---------------------------------------------------------------------------" << std::endl;
		}
		printResults(num_threads, MPI_COMM_WORLD);
#endif
		
    if(!myrank){
      tt[100*k+0]=ttt;
    }
  }

	// MPI_Finalize();
  // return 0;
  
  for(int i=p; myrank<i && i>=min_np; i=i>>1) proc_group++;
  MPI_Comm_split(MPI_COMM_WORLD, proc_group, myrank, &comm);
	
  { // smaller /2^k runs 
    int myrank_;
    MPI_Comm_rank(comm, &myrank_);
    double ttt;

    switch(dtype) {
			case 'd':
				ttt = time_sort<double>(N, comm,dist_type);
				break;
			case 'f':
				ttt = time_sort<float>(N, comm,dist_type);
				break;	
			case 'i':
        ttt = time_sort<int>(N, comm,dist_type);
        break;
      case 'l':
        ttt = time_sort<long>(N, comm,dist_type);
        break;
    };

    if(!myrank_){
      tt[100*k+proc_group]=ttt;
    }
  }

	int num_groups=0;
	if (!myrank) num_groups = proc_group;
	MPI_Bcast(&num_groups, 1, MPI_INT, 0, MPI_COMM_WORLD);

	for(size_t i = 0; i < num_groups; ++i)
	{
		MPI_Barrier(MPI_COMM_WORLD);
		if (proc_group == i) {
			MPI_Barrier(comm);
			printResults(num_threads, comm);
			MPI_Barrier(comm);
		}
	}

  MPI_Barrier(MPI_COMM_WORLD);

  std::vector<double> tt_glb(10000);
  MPI_Reduce(&tt[0], &tt_glb[0], 10000, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
  if(!myrank){
    std::cout<<"\nSort - Weak Scaling:\n";
    for(int i=0;i<proc_group;i++){
      int np=p;
      if(i>0) np=(p>>(i-1))-(p>>i);
      std::cout<<"\t\t\tP = "<<np<<"\t\t";
      // for(int k=0;k<=log_N;k++)
        std::cout<<tt_glb[100*k+i]<<' ';
      std::cout<<'\n';
    }
  }

  // Shut down MPI 
  MPI_Finalize();
   return 0;
 }

void printResults(int num_threads, MPI_Comm comm) {
	int myrank, p, simd_width=0;
	MPI_Comm_size(comm, &p);
	MPI_Comm_rank(comm, &myrank);
#ifdef SIMD_MERGE
	simd_width=SIMD_MERGE;
#endif
		// reduce results
		double t, meanV, minV, maxV;
		
#ifdef _PROFILE_SORT		
		if (!myrank) {
			// std::cout << std::endl;
			// std::cout << "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~" << std::endl;
			std::cout <<  p << " tasks : " << num_threads << " threads " << simd_width << " SIMD_WIDTH " << std::endl;
			// std::cout << "===========================================================================" << std::endl;
		}
		t = total_sort.seconds; 			getStats(t, &meanV, &minV, &maxV, comm);
		if (!myrank) {
			std::cout << "Total sort time   \t\t\t" << meanV << "\t" << minV << "\t" << maxV <<  std::endl;
			// std::cout << "----------------------------------------------------------------------" << std::endl;
		}
		t = seq_sort.seconds; 				getStats(t, &meanV, &minV, &maxV, comm);
		if (!myrank) {
			std::cout << "Sequential Sort   \t\t\t" << meanV << "\t" << minV << "\t" << maxV <<  std::endl;
		}
		t = sort_partitionw.seconds; 	getStats(t, &meanV, &minV, &maxV, comm);
		if (!myrank) {	
			std::cout << "partitionW        \t\t\t" << meanV << "\t" << minV << "\t" << maxV <<  std::endl; 
			// std::cout << "----------------------------------------------------------------------" << std::endl;
		}
#ifndef KWICK
		t = sample_sort_splitters.seconds; 				getStats(t, &meanV, &minV, &maxV, comm);
		if (!myrank) {	
			std::cout << "sort splitters    \t\t\t" << meanV << "\t" << minV << "\t" << maxV <<  std::endl; 
		}
		t = sample_prepare_scatter.seconds; 				getStats(t, &meanV, &minV, &maxV, comm);
		if (!myrank) {
			std::cout << "prepare scatter   \t\t\t" << meanV << "\t" << minV << "\t" << maxV <<  std::endl; 
		}
		t = sample_do_all2all.seconds; 				getStats(t, &meanV, &minV, &maxV, comm);	
		if (!myrank) {
			std::cout << "all2all           \t\t\t" << meanV << "\t" << minV << "\t" << maxV <<  std::endl; 
		}			 
#else
		t = hyper_compute_splitters.seconds; 				getStats(t, &meanV, &minV, &maxV, comm);		
		if (!myrank) {	
			std::cout << "compute splitters \t\t\t" << meanV << "\t" << minV << "\t" << maxV <<  std::endl; 
		}
		t = hyper_communicate.seconds; 				getStats(t, &meanV, &minV, &maxV, comm);	
		if (!myrank) {	
			std::cout << "exchange data     \t\t\t" << meanV << "\t" << minV << "\t" << maxV <<  std::endl; 
		}
		t = hyper_merge.seconds; 				getStats(t, &meanV, &minV, &maxV, comm);	
		if (!myrank) {	
			std::cout << "merge arrays      \t\t\t" << meanV << "\t" << minV << "\t" << maxV <<  std::endl; 
		}
		t = hyper_comm_split.seconds; 				getStats(t, &meanV, &minV, &maxV, comm);	
		if (!myrank) {	
			std::cout << "comm split        \t\t\t" << meanV << "\t" << minV << "\t" << maxV <<  std::endl; 
		}
    if (!myrank) {
      std::string num;
      std::stringstream mystream;
      mystream << total_bytes;
      num = mystream.str();
      int insertPosition = num.length() - 3;
      while (insertPosition > 0) {
        num.insert(insertPosition, ",");
        insertPosition-=3;
      }
      std::cout << "total comm        \t\t\t" << num << " bytes" <<  std::endl; 
    }

#endif
#endif
		if (!myrank) {			
			// std::cout << "---------------------------------------------------------------------------" << std::endl;
			std::cout << "" << std::endl;
		}			 
}

#define  FALSE          0       // Boolean false
#define  TRUE           1       // Boolean true
//===========================================================================
//=  Function to generate Zipf (power law) distributed random variables     =
//=    - Input: alpha and N                                                 =
//=    - Output: Returns with Zipf distributed random variable              =
//===========================================================================
int zipf(double alpha, int n, unsigned int *seedp)
{
  static int first = TRUE;      // Static first time flag
  static double c = 0;          // Normalization constant
  double z;                     // Uniform random number (0 < z < 1)
  double sum_prob;              // Sum of probabilities
  double zipf_value;            // Computed exponential value to be returned
  int    i;                     // Loop counter

  // Compute normalization constant on first call only
  if (first == TRUE)
  {
    for (i=1; i<=n; i++)
      c = c + (1.0 / pow((double) i, alpha));
    c = 1.0 / c;
    first = FALSE;
  }

  // Pull a uniform random number (0 < z < 1)
  do
  {
    z = rand_r(seedp)*(1.0/RAND_MAX);
  }
  while ((z == 0) || (z == 1));

	static std::vector<double> oopia;
	#pragma omp critical
	if(oopia.size()!=n){
		oopia.resize(n);
		for(int i=0;i<n;i++)
			oopia[i]=1.0/pow((double) i, alpha);
	}

  // Map z to the value
  sum_prob = 0;
  for (i=1; i<=n; i++)
  {
    sum_prob = sum_prob + c*oopia[i-1];
    if (sum_prob >= z)
    {
      zipf_value = i;
      break;
    }
  }

  // Assert that zipf_value is between 1 and N
  assert((zipf_value >=1) && (zipf_value <= n));

  return(zipf_value);
}
