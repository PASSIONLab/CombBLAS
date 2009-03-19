/****************************************************************/
/* Sequential and Parallel Sparse Matrix Multiplication Library */
/* version 2.2 --------------------------------------------------/
/* date: 05/27/2008 ---------------------------------------------/
/* author: Aydin Buluc (aydin@cs.ucsb.edu) ----------------------/
/* description: Parallel Test Application -----------------------/
/* tested classes: SparseCannon, SparseSumSych, SparseSumAsych --/
/****************************************************************/


#define BOOST_PTR_CONTAINER_NO_EXCEPTIONS 1


#include <sys/time.h>
#include <iostream>
#include <fstream>
#include <string>
#include <sstream>  // Required for stringstreams
#include <ctime>
#include <cmath>
#include "SparseTriplets.h"
#include "SparseDColumn.h"
#include "SparseSumSync.h"
//#include "SparseSumAsync.h"
//#include "SparseCannon.h"

using namespace std;
using namespace boost;

// Warning: Make sure you are using the correct NUMTYPE as your input files uses !
#define NUMTYPE double

		
#define NUMOFMATRICES 1
#define ITERATIONS 10

//#define SPSUMASYNC
#define SPSUMSYNC
//#define SPCANNON


int main(int argc, char* argv[])
{
	double finish;
	int provided, isMPIInit, myrank;

	if(argc < 2)
	{
		cerr << "The head directory should be passed as an argument" << endl;
		return -1;
	}
		
	MPI_Initialized(&isMPIInit);
	if(!isMPIInit)	// gasnet didn't initialize MPI, so go ahead.
		MPI_Init_thread( &argc, &argv, MPI_THREAD_MULTIPLE, &provided );

	/* Setup */
	MPI_Comm everyone;
	MPI_Comm_dup(MPI_COMM_WORLD, &everyone); 
	MPI_Comm_rank (everyone, &myrank);

	stringstream ss;
	string srank;
	ss << myrank;
	ss >> srank;

	string directory(argv[1]);
	directory = directory + "/proc";
	directory += srank; 
	
	string ifilename1 = "input1_";
	ifilename1 += srank;
	ifilename1 = directory+"/"+ifilename1;
	
	string ifilename2 = "input2_";
	ifilename2 += srank;
	ifilename2 = directory+"/"+ifilename2;

	ifstream input1(ifilename1.c_str());
	ifstream input2(ifilename2.c_str());
	

	#ifdef SPSUMSYNC
	{
		SparseSumSync<NUMTYPE> ColDistA(input1, everyone);
		SparseSumSync<NUMTYPE> ColDistB(input2, everyone);
	
		// multiply them
		SparseSumSync<NUMTYPE> ColDistC = ColDistA * ColDistB;
	
		if(myrank == 0)
			cout<<"Multiplications started"<<endl;	

		MPI_Barrier (everyone);
		double t1 = MPI_Wtime(); 	// initilize (wall-clock) timer
	
		for(int i=0;i<ITERATIONS; i++)
		{
			ColDistC = ColDistA*ColDistB;
		}
		MPI_Barrier (everyone);
		double t2 = MPI_Wtime(); 	

		if(myrank == 0)
		{
			cout<<"Multiplications finished"<<endl;	
			printf("%.6lf seconds elapsed\n", t2-t1);
		}

		input1.clear();
		input2.clear();
		input1.seekg(0,ios::beg);
		input2.seekg(0,ios::beg);

		/*
		string rfilename = "par_"; 
		rfilename += srank;
		rfilename = directory+"/"+rfilename;

		if(myrank == 0)
			cout<<"Writing output to disk"<<endl;
		ofstream outputr(rfilename.c_str()); 
		outputr << ColDistC; 
		*/
	
	}
	#endif

	#ifdef SPSUMASYNC
	{
		if ( provided != MPI_THREAD_MULTIPLE ) 
		{

			input1.close();
			input2.close();

			if(myrank == 1)
        		{	
				cout<< "MPI_THREAD_MULTIPLE is needed, not executing SparseSumAsync"<<endl;
			}
    		}
		else
		{
			SparseSumAsync<NUMTYPE> asynchA(input1, everyone);
			SparseSumAsync<NUMTYPE> asynchB(input2, everyone);
	
			SparseSumAsync<NUMTYPE> asynchC = asynchA * asynchB; 	// multiply them
	
			if(myrank == 0)
				cout<<"Multiplications started"<<endl;	

			MPI_Barrier (everyone);
			double t1 = MPI_Wtime(); 	// initilize (wall-clock) timer
	
			for(int i=0;i<ITERATIONS; i++)
			{
				asynchC = asynchA*asynchB;
			}

			MPI_Barrier (everyone);
			double t2 = MPI_Wtime(); 	

			if(myrank == 0)
			{
				cout<<"Multiplications finished"<<endl;	
				printf("%.6lf seconds elapsed\n", t2-t1);
			}

			input1.clear();
			input2.clear();
			input1.seekg(0,ios::beg);
			input2.seekg(0,ios::beg);
		}
	}
	#endif

	#ifdef SPCANNON
	{
		SparseCannon<NUMTYPE> cannonA(input1, everyone);
		SparseCannon<NUMTYPE> cannonB(input2, everyone);
	
		// multiply them
		SparseCannon<NUMTYPE> cannonC = cannonA * cannonB;
	
		if(myrank == 0)
			cout<<"Multiplications started"<<endl;	

		MPI_Barrier (everyone);
		double t1 = MPI_Wtime(); 	// initilize (wall-clock) timer

		for(int i=0;i<ITERATIONS; i++)
		{
			cannonC = cannonA*cannonB;
		}

		MPI_Barrier (everyone);
		double t2 = MPI_Wtime(); 	

		if(myrank == 0)
		{
			cout<<"Multiplications finished"<<endl;	
			printf("%.6lf seconds elapsed\n", t2-t1);
		}

		input1.close();
		input2.close();

		/*
		string rfilename = "can_"; 
		rfilename += srank;
		rfilename = directory+"/"+rfilename;

		if(myrank == 0)
			cout<<"Writing output to disk"<<endl;
		ofstream outputr(rfilename.c_str()); 
		outputr << cannonC; 
		*/
	}
	#endif

	MPI_Comm_free(&everyone);
	MPI_Finalize();
	
	return 0;
}

