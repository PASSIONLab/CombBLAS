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
#include "SpParVec.h"
#include "SpTuples.h"
#include "SpDCCols.h"
#include "SpParMPI2.h"

using namespace std;

//! Warning: Make sure you are using the correct NUMTYPE as your input files uses !
#define NUMTYPE float 

int main(int argc, char* argv[])
{
	MPI::Init();
	int nprocs = MPI::COMM_WORLD.Get_size();
	int myrank = MPI::COMM_WORLD.Get_rank();
	typedef PlusTimesSRing<double, double> PT;	

	if(argc < 4)
        {
		if(myrank == 0)
		{	
                	cout << "Usage: ./betwcent <BASEADDRESS> <K4APPROX> <BATCHSIZE>" << endl;
                	cout << "Example: ./betwcent Data/ 15 128" << endl;
                	cout << "Input file input.txt should be under <BASEADDRESS> in triples format" << endl;
                	cout << "<BATCHSIZE> should be a multiple of sqrt(p)" << endl;
 		}
		MPI::Finalize(); 
		return -1;
        }

	{
		int K4Approx = atoi(argv[2]);
		int batchSize = atoi(argv[3]);

		string directory(argv[1]);		
		string ifilename = "input.txt";
		ifilename = directory+"/"+ifilename;

		ifstream input(ifilename.c_str());
		MPI::COMM_WORLD.Barrier();
	
		// ABAB: Make a macro such as "PARTYPE(it,nt,seqtype)" that just typedefs this guy !
		typedef SpParMPI2 <int, bool, SpDCCols<int,bool> > PARBOOLMAT;
		typedef SpParMPI2 <int, int, SpDCCols<int,int> > PARINTMAT;

		PARBOOLMAT A;			// construct object
		A.ReadDistribute(input, 0);	// read it from file, note that we use the transpose of "input" data
		input.clear();
		input.close();
			
		SpParVec<int, double> bc(A.getcommgrid());	
		int nPasses = (int) pow(2.0, K4Approx);
		int numBatches = (int) ceil( static_cast<float>(nPasses)/ static_cast<float>(batchSize));
	
		// these get() calls are collective, should be called by all processors 
		int mA = A.getnrow(); 
		int nA = A.getncol();
		int nzA = A.getnnz();
	
		// get the number of batch vertices for submatrix
		int subBatchSize = batchSize / (A.getcommgrid())->GetGridCols();
		vector<int> batch(subBatchSize);

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
			PARINTMAT fringe = (A.SubsRefCol(batch)).ConvertNumericType<int, SpDCCols<int,int> >();
	
			int nrowperproc = mA / (A.getcommgrid())->GetGridCols();
			// copy(batch.begin(), batch.end(), ostream_iterator<int>(cout, " "));

			int m_f = fringe.getnrow(); 
			int n_f = fringe.getncol();
			int nz_f = fringe.getnnz();

			if (myrank == 0)
				cout << "Fringe has " << m_f << " rows and "<< n_f <<" columns and "<<  nz_f << " nonzeros" << endl;

			// Create nsp by setting (r,i)=1 for the ith root vertex with label r
			// Inially only the diagonal processors have any nonzeros (because we chose roots so)
			SpDCCols<int,int> * nsploc = new SpDCCols<int,int>();
			tuple<int, int, int> * mytuples = NULL;	
			if(A.getcommgrid()->GetRankInProcRow() == A.getcommgrid()->GetRankInProcCol())
			{
				mytuples = new tuple<int, int, int>[subBatchSize];
				for(int k =0; k<subBatchSize; ++k)
				{
					mytuples[k] = make_tuple(batch[k], k, 1);
				}
				nsploc->Create( subBatchSize, A.getlocalrows(), subBatchSize, mytuples);		
			}
			else
			{
				nsploc->Create( 0, A.getlocalrows(), subBatchSize, mytuples);		
			}
		
			PARINTMAT nsp(nsploc, A.getcommgrid());	// This parallel data structure HAS-A SpTuples
	
			int m_nsp = nsp.getnrow(); 
			int n_nsp = nsp.getncol();
			int nz_nsp = nsp.getnnz();

			if (myrank == 0)
				cout << "NSP has " << m_nsp << " rows and "<< n_nsp <<" columns and "<<  nz_nsp << " nonzeros" << endl;
	
			vector < void * > bfs;	// internally keeps track of depth
			typedef PlusTimesSRing<int, int> PTINT;		
			
			while( fringe.getnnz() > 0 )
			{
				nsp += fringe;
				bfs.push_back(new PARBOOLMAT(fringe.ConvertNumericType<bool, SpDCCols<int,bool> >() )); 

				fringe = (Mult_AnXBn<PTINT>(A, fringe)).ElementWiseMult(nsp, true);

			}
			m_nsp = nsp.getnrow(); 
			n_nsp = nsp.getncol();
			nz_nsp = nsp.getnnz();

			if (myrank == 0)
				cout << "NSP has " << m_nsp << " rows and "<< n_nsp <<" columns and "<<  nz_nsp << " nonzeros" << endl;

	
			/*
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
			*/ 
		}

	}	

	// make sure the destructors for all objects are called before MPI::Finalize()
	MPI::Finalize();	
	return 0;
}

