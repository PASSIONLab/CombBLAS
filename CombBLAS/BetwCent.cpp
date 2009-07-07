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
#include "DenseParMat.h"
#include "DenseParVec.h"

using namespace std;

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
			cout << "Because <BATCHSIZE> is for the overall matrix, whereas <K4APPROX> is per processor " << endl;
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
		typedef SpParMPI2 <int, double, SpDCCols<int,double> > PARDOUBLEMAT;

		PARBOOLMAT A, AT;			// construct object
		A.ReadDistribute(input, 0);	// read it from file, note that we use the transpose of "input" data
		AT = A;
		AT.Transpose();

		input.clear();
		input.close();
			
		int nPasses = (int) pow(2.0, K4Approx);
		int numBatches = (int) ceil( static_cast<float>(nPasses)/ static_cast<float>(batchSize));
	
		// get the number of batch vertices for submatrix
		int subBatchSize = batchSize / (A.getcommgrid())->GetGridCols();
		if(batchSize % (A.getcommgrid())->GetGridCols() > 0 && myrank == 0)
			cout << "*** Please make batchsize divisible by the grid dimensions (r and s) ***" << endl;

		vector<int> candidates;
		if (myrank == 0)
			cout << "Batch processing will occur " << numBatches << " times, each processing " << batchSize << " vertices (overall)" << endl;

		// Only consider non-isolated vertices
		int vertices = 0;
		int vrtxid = 0; 
		int nlocpass = nPasses / (A.getcommgrid())->GetGridCols();
		while(vertices < nlocpass)
		{
			vector<int> single;
			vector<int> empty;
			single.push_back(vrtxid);
			int locnnz = ((A.seq())(empty,single)).getnnz();
			int totnnz;
			(A.getcommgrid())->GetColWorld().Allreduce( &locnnz, &totnnz, 1, MPI_INT, MPI::SUM);
					
			if(totnnz > 0)
			{
				candidates.push_back(vrtxid);
				++vertices;
			}
			++vrtxid;
		}

		double t1 = MPI_Wtime();
		vector<int> batch(subBatchSize);
		for(int i=0; i< numBatches; ++i)
		{
			for(int j=0; j< subBatchSize; ++j)
			{
				batch[j] = candidates[i*subBatchSize + j];
			}
			
			PARINTMAT fringe = (A.SubsRefCol(batch)).ConvertNumericType<int, SpDCCols<int,int> >();
			fringe.PrintInfo();

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
				
			vector < PARBOOLMAT * > bfs;	// internally keeps track of depth
			typedef PlusTimesSRing<bool, int> PTBOOLINT;	
			typedef PlusTimesSRing<bool, double> PTBOOLDOUBLE;	
	
			while( fringe.getnnz() > 0 )
			{
				nsp += fringe;
				PARBOOLMAT * level = new PARBOOLMAT(fringe.ConvertNumericType<bool, SpDCCols<int,bool> >() ); 
				bfs.push_back(level);

				fringe = (Mult_AnXBn<PTBOOLINT>(A, fringe));
				fringe.ElementWiseMult(nsp, true);	
			}

			// Apply the unary function 1/x to every element in the matrix
			// 1/x works because no explicit zeros are stored in the sparse matrix nsp
			PARDOUBLEMAT nspInv = nsp.ConvertNumericType<double, SpDCCols<int,double> >();
			nspInv.Apply(bind1st(divides<double>(), 1));

			double ** bculocal = SpHelper::allocate2D<double>(fringe.getlocalrows(), fringe.getlocalcols());
			for(int r=0; r< fringe.getlocalrows(); ++r)
				fill_n(bculocal[r], fringe.getlocalcols(), 1.0);

			DenseParMat<int, double> bcu(bculocal, A.getcommgrid(), fringe.getlocalrows(), fringe.getlocalcols() );

			// BC update for all vertices except the sources
			for(int j = bfs.size()-1; j > 0; --j)
			{
				PARDOUBLEMAT w = EWiseMult( *bfs[j], nspInv, false);
				w.ElementWiseScale(bcu);

				PARDOUBLEMAT product = Mult_AnXBn<PTBOOLDOUBLE>(AT,w);
				product = EWiseMult(product, *bfs[j-1], false);
				product = EWiseMult(product, nsp, false);		

				bcu += product;
			}
			for(int j=0; j < bfs.size(); ++j)
			{
				delete bfs[j];
			}
		
			// Accumulate bcu to bc
		}
		double t2=MPI_Wtime();
		double TEPS = (nPasses * static_cast<float>(A.getnnz())) / (t2-t1);
		if( myrank == 0)
		{
			cout<<"Computation finished"<<endl;	
			fprintf(stdout, "%.6lf seconds elapsed for %d starting vertices\n", t2-t1, nPasses);
			fprintf(stdout, "TEPS score is: %.6lf\n", TEPS);
		}
	}	

	// make sure the destructors for all objects are called before MPI::Finalize()
	MPI::Finalize();	
	return 0;
}

