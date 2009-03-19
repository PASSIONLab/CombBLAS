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
#include "SparseMatrix.h"
#include "SparseTriplets.h"
#include "SparseDColumn.h"
#include "SparseOneSidedMPI.h"

using namespace std;
using namespace boost;

//! Warning: Make sure you are using the correct NUMTYPE as your input files uses !
#define NUMTYPE double
#define ITERATIONS 10

int main(int argc, char* argv[])
{
	int  myrank, nprocs;

    	MPI_Init(&argc, &argv);
	MPI_Comm wholegrid;
	MPI_Comm_dup(MPI_COMM_WORLD, &wholegrid); 
    	MPI_Comm_size(wholegrid, &nprocs);
    	MPI_Comm_rank(wholegrid, &myrank);

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

	MPI_Barrier (wholegrid);
	{
		SparseOneSidedMPI<NUMTYPE> A(input1, wholegrid);
		SparseOneSidedMPI<NUMTYPE> B(input2, wholegrid);

		input1.clear();
		input2.clear();
		input1.close();
		input2.close();

		// multiply them to warm-up caches
		SparseOneSidedMPI<NUMTYPE> C = A * B;

		if( myrank == 0)
			cout<<"Multiplications started"<<endl;	

		MPI_Barrier (wholegrid);
		double t1 = MPI_Wtime();	// start timer (actual wall-clock time)
		
		for(int i=0;i<ITERATIONS; i++)
		{
			SparseOneSidedMPI<NUMTYPE> C = A * B;
		}
		
		MPI_Barrier (wholegrid);
        	double t2=MPI_Wtime();
		
		if( myrank == 0)
		{
			cout<<"Multiplications finished"<<endl;	
			fprintf(stdout, "%.6lf seconds elapsed for %d iterations\n", t2-t1, ITERATIONS);
		}

		ITYPE mA = A.getrows(); 
		ITYPE nA = A.getcols();
		ITYPE nzA = A.getnnz();
		ITYPE mB = B.getrows();
		ITYPE nB = B.getcols();
		ITYPE nzB = B.getnnz();
		ITYPE mC = C.getrows();
		ITYPE nC = C.getcols();
		ITYPE nzC = C.getnnz();
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
	MPI_Comm_free(&wholegrid);
	MPI_Finalize();
	
	return 0;
}

