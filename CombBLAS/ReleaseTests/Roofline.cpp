#include <mpi.h>
#include <sys/time.h> 
#include <iostream>
#include <functional>
#include <algorithm>
#include <vector>
#include <sstream>
#include "../CombBLAS.h"
#include "../Applications/TwitterEdge.h"

using namespace std;

// One edge = 16 bytes (rounded up from 11)
// One parent = 8 bytes
#define INC 256
#define L1 2048	// maximum entries of edges+parents combined to fit 32 KB
#define REPEAT 100

struct twitter_mult : public std::binary_function<ParentType, TwitterEdge, ParentType>
{
  ParentType operator()(const ParentType & arg1, const TwitterEdge & arg2) const
  {
	time_t now = time(0);
        struct tm * timeinfo = localtime( &now);
	timeinfo->tm_mon = timeinfo->tm_mon-1;
	time_t monthago = mktime(timeinfo);
	if(arg2.isRetwitter() && arg2.TweetWithinInterval(monthago, now))       // T1 is of type edges for BFS
		return arg1;
	else
		return ParentType();    // null-type parent id
  }
};


int main(int argc, char* argv[])
{
	MPI::Init(argc, argv);
	int nprocs = MPI::COMM_WORLD.Get_size();
	int myrank = MPI::COMM_WORLD.Get_rank();

	{
		int64_t len = INC;
		time_t now;
		time ( &now );
		TwitterEdge twe(4, 1, now);	// 4 retweets, latest now, following

		while(len < L1)
		{
			//  FullyDistVec(IT globallen, NT initval)
			FullyDistVec<int64_t,TwitterEdge> tvec(nprocs * len, twe);
			FullyDistVec<int64_t,ParentType> pvec;
			pvec.iota(nprocs * len, ParentType());
	
			MPI::COMM_WORLD.Barrier();
			double t1 = MPI::Wtime(); 	// initilize (wall-clock) timer

			for(int i=0; i< REPEAT; ++i)
				pvec.EWiseApply(tvec, twitter_mult());


			MPI::COMM_WORLD.Barrier();
			double t2 = MPI::Wtime(); 	

			if(myrank == 0)
			{
				cout<<"EWiseApply Iterations finished"<<endl;	
				double time = t2-t1;
				double teps = (nprocs*len*REPEAT) / (time * 1000000);
				printf("%.6lf seconds elapsed for %d iterations on vector of length %lld\n", time, REPEAT, nprocs*len);
				printf("%.6lf million TEPS per second\n", teps);
			}
			len += INC;
		}
	}
	MPI::Finalize();
	return 0;
}
