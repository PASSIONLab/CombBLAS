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
#include <boost/ptr_container/ptr_vector.hpp> 	// ABAB: use vector<void*> instead !
#include "SparseMatrix.h"
#include "SpParVec.h"
#include "SpParMatrix.h"
#include "SparseOneSidedMPI.h"

using namespace std;
using namespace boost;

//! Warning: Make sure you are using the correct NUMTYPE as your input files uses !
#define NUMTYPE float 

int main(int argc, char* argv[])
{
	int  myrank, nprocs;
	MPI::Init();
	MPI_Comm wholegrid;
	MPI_Comm_dup(MPI_COMM_WORLD, &wholegrid); 
	MPI_Comm_size(wholegrid, &nprocs);
	MPI_Comm_rank(wholegrid, &myrank);

	if(argc < 4)
        {
		if(myrank == 0)
		{	
                	cout << "Usage: ./testseq <BASEADDRESS> <K4APPROX> <BATCHSIZE>" << endl;
                	cout << "Example: ./testseq Data/LOGRMAT17-TRANSPOSED/ 15 128" << endl;
                	cout << "Input files input1_p should be under <BASEADDRESS>/proc_p/ " << endl;
 		}
		MPI::Finalize(); 
		return -1;
        }
	int K4Approx = atoi(argv[2]);
	int batchSize = atoi(argv[3]);


	stringstream ss1, ss2;
	string rank, nodes;
	ss1 << myrank;
	ss1 >> rank;
	ss2 << nprocs;
	ss2 >> nodes;

	string directory(argv[1]);		
	directory = directory + "/p";
	directory += nodes;
	directory = directory + "/proc";
	directory += rank; 
	
	string ifilename1 = "input1_";
	ifilename1 += rank;
	ifilename1 = directory+"/"+ifilename1;

	ifstream input1(ifilename1.c_str());

	MPI_Barrier (wholegrid);

	SpParMatrix<NUMTYPE> * A = new SparseOneSidedMPI<NUMTYPE>(input1, wholegrid);
	input1.clear();
	input1.close();
		
	SpParVec<NUMTYPE> * bc = new SpParVec<NUMTYPE>(A->getcommgrid());
	int nPasses = (int) pow(2.0, K4Approx);
	int numBatches = (int) ceil( static_cast<float>(nPasses)/ static_cast<float>(batchSize));

	// these get() calls are collective, should be called by all processors 
	ITYPE mA = A->getrows(); 
	ITYPE nA = A->getcols();
	ITYPE nzA = A->getnnz();

	// get the number of batch vertices for submatrix
	ITYPE subBatchSize = batchSize / (A->getcommgrid())->grcol;
	vector<ITYPE> batch(subBatchSize);

	if (myrank == 0)
	{
		cout << "A has " << mA << " rows and "<< nA <<" columns and "<<  nzA << " nonzeros" << endl;
		cout << "Batch processing will occur " << numBatches << " times, each processing " << batchSize << " vertices" << endl;
		cout << "SubBatch size is: " << subBatchSize << endl;
	}

	for(int i=0; i< numBatches; ++i)
	{
		for(int j=0; j< subBatchSize; ++j)
		{
			batch[j] = i*subBatchSize + j;
		}
		SpParMatrix<NUMTYPE> * fringe = A->SubsRefCol(batch);

		// Create nsp by setting (r,i)=1 for the ith root vertex with label r
		// Inially only the diagonal processors have any nonzeros (because we chose roots so)
		int rowrank, colrank;
		MPI_Comm_rank(A->getcommgrid()->rowWorld, &rowrank);
		MPI_Comm_rank(A->getcommgrid()->colWorld, &colrank);

		shared_ptr< SparseDColumn<NUMTYPE> > nsplocal;
		if(rowrank == colrank)
		{
			tuple<ITYPE, ITYPE, NUMTYPE> * mytuples = new tuple<ITYPE, ITYPE, NUMTYPE>[subBatchSize];
			for(int k =0; k<subBatchSize; ++k)
			{
				mytuples[k] = tuple<ITYPE, ITYPE, NUMTYPE>(batch[k], k, 1.0);
			}

			SparseTriplets<NUMTYPE> triples(subBatchSize, A->getlocalrows(), subBatchSize, mytuples);
			nsplocal.reset(new SparseDColumn<NUMTYPE>(triples, false));

		}
		else
		{
			SparseTriplets<NUMTYPE> triples(0, A->getlocalrows(), subBatchSize);
			nsplocal.reset(new SparseDColumn<NUMTYPE>(triples, false));
		}
		
		SpParMatrix<NUMTYPE> * nsp = new SparseOneSidedMPI<NUMTYPE>(nsplocal, A->getcommgrid());
		
		
		int depth = 0;
		ptr_vector < SparseDColumn<NUMTYPE> > bfs;
			
		while( fringe->getnnz() > 0 )
		{
			(*nsp) += (*fringe);
			 
			//bfs.push_back(new SparseDColumn<NUMTYPE>(*(fringe))); 

			depth++;
		}
	
		string rfilename = "fridge_"; 
		rfilename += rank;
		rfilename = directory+"/"+rfilename;
		
		if(myrank == 0)
			cout<<"Writing output to disk"<<endl;
		ofstream outputr(rfilename.c_str()); 
		outputr << (*fringe);	 
		if(myrank == 0)
			cout <<"Wrote to disk" << endl;

		delete fringe;

		break; 
	}

	MPI_Comm_free(&wholegrid);
	delete A;
	delete bc;

	MPI::Finalize();	
	return 0;
}

