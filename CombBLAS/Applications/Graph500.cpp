#include <mpi.h>
#include <sys/time.h> 
#include <iostream>
#include <functional>
#include <algorithm>
#include <vector>
#include <sstream>
#include <stdint.h>
#ifdef NOTR1
        #include <boost/tr1/memory.hpp>
#else
        #include <tr1/memory>
#endif
#include "../SpParVec.h"
#include "../SpTuples.h"
#include "../SpDCCols.h"
#include "../SpParMat.h"
#include "../DenseParMat.h"
#include "../DenseParVec.h"
#include "../ParFriends.h"
#include "../DistEdgeList.h"


using namespace std;

// 64-bit floor(log2(x)) function 
// note: least significant bit is the "zeroth" bit
// pre: v > 0
unsigned int highestbitset(uint64_t v)
{
	// b in binary is {10,1100, 11110000, 1111111100000000 ...}  
	const uint64_t b[] = {0x2ULL, 0xCULL, 0xF0ULL, 0xFF00ULL, 0xFFFF0000ULL, 0xFFFFFFFF00000000ULL};
	const unsigned int S[] = {1, 2, 4, 8, 16, 32};
	int i;

	unsigned int r = 0; // result of log2(v) will go here
	for (i = 5; i >= 0; i--) 
	{
		if (v & b[i])	// highestbitset is on the left half (i.e. v > S[i] for sure)
		{
			v >>= S[i];
			r |= S[i];
		} 
	}
	return r;
}

int main(int argc, char* argv[])
{
	MPI::Init(argc, argv);
	int nprocs = MPI::COMM_WORLD.Get_size();
	int myrank = MPI::COMM_WORLD.Get_rank();

	bool NOINPUT = true;
	if(argc < 2)
	{
		if(myrank == 0)
		{
			cout << "Usage: ./Graph500 <Available RAM in MB (per core)>" << endl;
			cout << "Example: ./Graph500 1024" << endl;
		}
		MPI::Finalize(); 
		return -1;
	}		
	else if(argc == 3)
	{
		NOINPUT = false;	// we indeed do have to read from input
	}		
	{
		typedef SelectMaxSRing<bool, int64_t> SR;	
		typedef SpParMat < int64_t, bool, SpDCCols<int64_t,bool> > PSpMat_Bool;
		typedef SpParMat < int64_t, int, SpDCCols<int64_t,int> > PSpMat_Int;
		typedef SpParMat < int64_t, int64_t, SpDCCols<int64_t,int64_t> > PSpMat_Int64;

		// calculate the problem size that can be solved
		// number of nonzero columns are at most the matrix dimension (for small p)
		// for large p, though, nzc = nnz since each subcolumn will have a single nonzero 
		// so assume (1+8+8+8)*nedges for the uint64 case and (1+4+4+4)*nedges for uint32
		uint64_t raminbytes = static_cast<uint64_t>(atoi(argv[1])) * 1024 * 1024;	
		uint64_t peredge = 1+3*sizeof(int64_t);
		uint64_t maxnedges = raminbytes / peredge;
		uint64_t maxvertices = maxnedges / 32;	
		unsigned maxscale = highestbitset(maxvertices * nprocs);

		string name;
		unsigned scale;
		if(maxscale > 36)	// at least 37 so it fits comfortably along with vectors 
		{
			name = "Medium";	
			scale = 36;
		}
		else if(maxscale > 32)
		{
			name = "Small";
			scale = 32;
		}
		else if(maxscale > 29)
		{
			name = "Mini";
			scale = 29;
		}
		else if(maxscale > 26)
		{
			name = "Toy";
			scale = 26;
		}
		else
		{
			name = "Debug";
			scale = 17;	// fits even to single processor
		}

		ostringstream outs;
		outs << "Max scale allowed : " << maxscale << endl;
		outs << "Using the " << name << " problem" << endl;
		SpParHelper::Print(outs.str());

		// Declare objects
		PSpMat_Bool A;	
		DenseParVec<int64_t, int64_t> degrees;

		if(NOINPUT)
		{
			// this is an undirected graph, so A*x does indeed BFS
 			double initiator[4] = {.57, .19, .19, .05};

			DistEdgeList<int64_t> DEL;
			DEL.GenGraph500Data(initiator, scale, 16 * ((int64_t) std::pow(2.0, (double) scale)) / nprocs );
			PermEdges<int64_t>(DEL);
			RenameVertices<int64_t>(DEL);

			PSpMat_Int64 * G = new PSpMat_Int64(DEL, false);	 // conversion from distributed edge list, keep self-loops
			degrees = G->Reduce(Column, plus<int64_t>(), 0); 
			delete G;

			// Start Kernel #1
			MPI::COMM_WORLD.Barrier();
			double t1 = MPI_Wtime();

			A = PSpMat_Bool(DEL);	// remove self loops and duplicates (since this is of type boolean)
			PSpMat_Bool AT = A;
			AT.Transpose();
			A += AT;
			
			MPI::COMM_WORLD.Barrier();
			double t2=MPI_Wtime();
			
			if(myrank == 0)
				fprintf(stdout, "%.6lf seconds elapsed for Kernel #1\n", t2-t1);
		}
		else
		{
			ifstream input(argv[2]);
			A.ReadDistribute(input, 0);	// read it from file
			SpParHelper::Print("Read input");
			PSpMat_Bool AT = A;
			AT.Transpose();

			// boolean addition is practically a "logical or", 
			// therefore this doesn't destruct any links
			A += AT;	// symmetricize
		}
				
		A.PrintInfo();

		float balance = A.LoadImbalance();
		outs << "Load balance: " << balance << endl;
		SpParHelper::Print(outs.str());

		PSpMat_Int * AInt = new PSpMat_Int(A); 
		DenseParVec<int64_t, int> * ColSums = new DenseParVec<int64_t, int> (AInt->Reduce(Column, plus<int>(), false)); 
		DenseParVec<int64_t, int64_t> Cands = ColSums->FindInds(bind2nd(greater<int>(), 1));	// only the indices of non-isolated vertices
		delete AInt;
		delete ColSums;
			
		Cands.PrintInfo("Candidates array");
		DenseParVec<int64_t,int64_t> First64(A.getcommgrid(), -1);
		Cands.RandPerm();
		Cands.PrintInfo("Candidates array (permuted)");
		First64.iota(64, 0);			
		Cands = Cands(First64);		
		Cands.PrintInfo("First 64 of candidates (randomly chosen) array");
		
		double MTEPS[64]; double INVMTEPS[64];
		for(int i=0; i<64; ++i)
		{
			// DenseParVec ( shared_ptr<CommGrid> grid, IT locallength, NT initval, NT id);
			DenseParVec<int64_t, int64_t> parents ( A.getcommgrid(), A.getlocalcols(), (int64_t) -1, (int64_t) -1);	// identity is -1
			SpParVec<int64_t, int64_t> fringe(A.getcommgrid(), A.getlocalcols());	// numerical values are stored 0-based

			ostringstream outs;
			outs << "Starting vertex id: " << Cands[i] << endl;
			SpParHelper::Print(outs.str());
			fringe.SetElement(Cands[i], Cands[i]);
			int iterations = 0;

			MPI::COMM_WORLD.Barrier();
			double t1 = MPI_Wtime();
			while(fringe.getnnz() > 0)
			{
				fringe.setNumToInd();
				//fringe.PrintInfo("fringe before SpMV");
				fringe = SpMV<SR>(A, fringe);	// SpMV with sparse vector
				//fringe.PrintInfo("fringe after SpMV");
				fringe = EWiseMult(fringe, parents, true, (int64_t) -1);	// clean-up vertices that already has parents 
				//fringe.PrintInfo("fringe after cleanup");
				parents += fringe;
				//SpParHelper::Print("Iteration finished\n");
				iterations++;
			}
			MPI::COMM_WORLD.Barrier();
			double t2 = MPI_Wtime();

			SpParVec<int64_t, int64_t> parentsp = parents.Find(bind2nd(greater<int64_t>(), -1));
			parentsp.Apply(set<int64_t>(1));
			int64_t nedges = EWiseMult(parentsp, degrees, false, (int64_t) 0).Reduce(plus<int64_t>(), (int64_t) 0);
	
			ostringstream outnew;
			outnew << "Number iterations: " << iterations << endl;
			outnew << "Number of vertices found: " << parentsp.Reduce(plus<int64_t>(), (int64_t) 0) << endl; 
			outnew << "Number of edges traversed: " << nedges << endl;
			outnew << "BFS time: " << t2-t1 << " seconds" << endl;
			outnew << "MTEPS: " << static_cast<double>(nedges) / (t2-t1) / 1000000.0 << endl;
			MTEPS[i] = static_cast<double>(nedges) / (t2-t1) / 1000000.0;
			SpParHelper::Print(outnew.str());
			parents.PrintInfo("parents after BFS");	
		}
		transform(MTEPS, MTEPS+64, INVMTEPS, safemultinv<double>()); 	
		double hteps = 64.0 / accumulate(INVMTEPS, INVMTEPS+64, 0.0);
		ostringstream os;
		os << "Harmonic mean of TEPS: " << hteps << endl;
		SpParHelper::Print(os.str());
	}
	MPI::Finalize();
	return 0;
}

