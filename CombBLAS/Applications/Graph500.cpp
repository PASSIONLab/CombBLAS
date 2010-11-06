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
			scale = 12;	// fits even to single processor
		}

		ostringstream outs;
		outs << "Max scale allowed : " << maxscale << endl;
		outs << "Using the " << name << " problem" << endl;
		SpParHelper::Print(outs.str());

		// Declare objects
		SpParMat<int64_t, bool, SpDCCols<int64_t, bool> > A;	
		DenseParVec<int64_t, int64_t> x;

		if(NOINPUT)
		{
			// this is an undirected graph, so A*x does indeed BFS
 			double initiator[4] = {.57, .19, .19, .05};

			DistEdgeList<int64_t> DEL;
			DEL.GenGraph500Data(initiator, scale, 16 * ((int64_t) std::pow(2.0, (double) scale)) / nprocs );
			PermEdges<int64_t>(DEL);
			RenameVertices<int64_t>(DEL);

			A = SpParMat<int64_t, bool, SpDCCols<int64_t, bool> > (DEL);	 // conversion from distributed edge list
			PSpMat_Bool AT = A;
			AT.Transpose();
			A += AT;
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

		// Reduce on a boolean matrix would return a boolean vector, not possible to sum along
		PSpMat_Int * AInt = new PSpMat_Int(A);
		DenseParVec<int64_t, int> ColSums = AInt->Reduce(Column, plus<int>(), 0); 
		DenseParVec<int64_t, int64_t> Cands = ColSums.FindInds(bind2nd(greater<int>(), 2));	// only the indices of connected vertices
		Cands.PrintInfo("Candidates array");
		delete AInt;	// save memory	

		DenseParVec<int64_t,int64_t> First64(A.getcommgrid(), -1);
		Cands.RandPerm();
		Cands.PrintInfo("Candidates array (permuted)");
		First64.iota(64, 0);			
		Cands = Cands(First64);		
		Cands.DebugPrint();
		Cands.PrintInfo("First 64 of candidates (randomly chosen) array");

		for(int i=0; i<64; ++i)
		{
			// DenseParVec ( shared_ptr<CommGrid> grid, IT locallength, NT initval, NT id);
			DenseParVec<int64_t, int64_t> parents ( A.getcommgrid(), A.getlocalcols(), (int64_t) -1, (int64_t) -1);	// identity is -1
			DenseParVec<int64_t, int> levels;
			int64_t level = 1;
			SpParVec<int64_t, int64_t> fringe(A.getcommgrid(), A.getlocalcols());	// numerical values are stored 0-based

			ostringstream outs;
			outs << "Starting vertex id: " << Cands[i] << endl;
			SpParHelper::Print(outs.str());
			fringe.SetElement(Cands[i], Cands[i]);	
			while(fringe.getnnz() > 0)
			{
				fringe.setNumToInd();
				fringe.PrintInfo("fringe before SpMV");

				fringe = SpMV<SR>(A, fringe);	// SpMV with sparse vector
				fringe.PrintInfo("fringe after SpMV");
				fringe = EWiseMult(fringe, parents, true, (int64_t) -1);	// clean-up vertices that already has parents 
				fringe.PrintInfo("fringe after cleanup");

				parents += fringe;

				// following steps are only for validation later
				// SpParVec<int64_t, int> thislevel(fringe);
				// thislevel.Apply(set<int>(level++));	
				// levels += thislevel;
				SpParHelper::Print("Iteration finished\n");
			}
			parents.PrintInfo("parents after BFS");	
		}
	}
	MPI::Finalize();
	return 0;
}

