/****************************************************************/
/* Sequential and Parallel Sparse Matrix Multiplication Library */
/* version 2.3 --------------------------------------------------/
/* date: 01/18/2009 ---------------------------------------------/
/* author: Aydin Buluc (aydin@cs.ucsb.edu) ----------------------/
/****************************************************************/

#include <mpi.h>
#include <sys/time.h>
#include <iostream>
#include <fstream>
#include <string>
#include <sstream>  // Required for stringstreams
#include <ctime>
#include <cmath>
#include "SpMat.h"
#include "SpTuples.h"
#include "SpDCCols.h"
#include "SpParMPI2.h"

using namespace std;

//! Warning: Make sure you are using the correct NUMT as your input files uses !
#define INDT unsigned
#define NUMT double
#define SEQM SpDCCols<INDT, NUMT>
#define ITERATIONS 10

int main(int argc, char* argv[])
{
	MPI::Init();
	int nproc = MPI::COMM_WORLD.Get_size();
	int myrank = MPI::COMM_WORLD.Get_rank();
    	
	stringstream ss1, ss2;
	string rank, nodes;
	ss1 << myrank;
	ss1 >> rank;
	ss2 << nprocs;
	ss2 >> nodes;

	string directory(BASEADDRESS);		
	directory = directory + "/p";
	directory += nodes;
	directory = directory + "/proc";
	directory += rank; 
	
	string ifilename1 = "input1_";
	ifilename1 += rank;
	ifilename1 = directory+"/"+ifilename1;

	string ifilename2 = "input2_";
	ifilename2 += rank;
	ifilename2 = directory+"/"+ifilename2;

	ifstream input1(ifilename1.c_str());
	ifstream input2(ifilename2.c_str());

	MPI::COMM_WORLD.Barrier();
	{
		SpParMPI2<INDT, NUMT, SEQM> A(input1, MPI::COMM_WORLD);
		SpParMPI2<INDT, NUMT, SEQM> B(input2, MPI::COMM_WORLD);

		input1.clear();
		input2.clear();
		input1.close();
		input2.close();

		// multiply them to warm-up caches
		SpParMPI2< INDT, NUMT, SEQM > C = A * B;

		if( myrank == 0)
			cout<<"Multiplications started"<<endl;	

		MPI::COMM_WORLD.Barrier();
		double t1 = MPI::Wtime();	// start timer (actual wall-clock time)
		
		for(int i=0;i<ITERATIONS; i++)
		{
			// This is a different C (as it is in a different scope)
			SpParMPI2< INDT, NUMT, SEQM > C = A * B;
		}
		
		MPI::COMM_WORLD.Barrier();
        	double t2=MPI::Wtime();
		
		if( myrank == 0)
		{
			cout<<"Multiplications finished"<<endl;	
			fprintf(stdout, "%.6lf seconds elapsed for %d iterations\n", t2-t1, ITERATIONS);
		}

		INDT mA = A.getnrow(); 
		INDT nA = A.getncol();
		INDT nzA = A.getnnz();
		INDT mB = B.getnrow();
		INDT nB = B.getncol();
		INDT nzB = B.getnnz();
		INDT mC = C.getnrow();
		INDT nC = C.getncol();
		INDT nzC = C.getnnz();
		if (myrank == 0)
		{
			cout <<"A has " << mA << " rows and "<< nA <<" columns and "<<  nzA << " nonzeros" << endl;
			cout <<"B has " << mB << " rows and "<< nB <<" columns and "<<  nzB << " nonzeros" << endl;	
			cout <<"C has " << mC << " rows and "<< nC <<" columns and "<<  nzC << " nonzeros" << endl;
		}
		
		string rfilename = "onesided_"; 
		rfilename += rank;
		rfilename = directory+"/"+rfilename;
		
		if(myrank == 0)
			cout<<"Writing output to disk"<<endl;
		ofstream outputr(rfilename.c_str()); 
		outputr << C;	 
		if(myrank == 0)
			cout <<"Wrote to disk" << endl;
	}
	MPI::Finalize();
	
	return 0;
}

