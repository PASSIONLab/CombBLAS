#include <mpi.h>
#include <sys/time.h> 
#include <iostream>
#include <functional>
#include <algorithm>
#include <vector>
#include <sstream>
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


using namespace std;

// 32-bit floor(log2(x)) function 
// note: least significant bit is the "zeroth" bit
// pre: v > 0
unsigned int highestbitset(unsigned int v)
{
	// b in binary is {10,1100, 11110000, 1111111100000000 ...}  
	const unsigned int b[] = {0x2, 0xC, 0xF0, 0xFF00, 0xFFFF0000};
	const unsigned int S[] = {1, 2, 4, 8, 16};
	int i;

	unsigned int r = 0; 
	for (i = 4; i >= 0; i--) 
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
	{
		typedef bool NM;
		typedef unsigned NV;
		typedef unsigned IT;
		typedef SelectMaxSRing<NM, NV> SR;	
		typedef SpParMat < IT, NV, SpDCCols<IT,NV> > PSpMat;

		// Declare objects
		PSpMat AT;	
		DenseParVec<int, double> x;

		// calculate the problem size that can be solved
		// number of nonzero columns are at most the matrix dimension (for small p)
		// for large p, though, nzc = nnz since each subcolumn will have a single nonzero 
		// so assume (1+8+8+8)*nedges for the uint64 case and (1+4+4+4)*nedges for uint32
		unsigned raminbytes = atoi(argv[1]) * 1024 * 1024;	
		unsigned peredge = 1+3*sizeof(IT);
		unsigned maxnedges = raminbytes / peredge;
		unsigned maxvertices = maxnedges / 16;
		unsigned maxscale = highestbitset(maxvertices) * nprocs;

		string name;
		unsigned scale;
		if(scale > 36)	// at least 37 so it fits comfortably along with vectors 
		{
			name = "Medium";	
			scale = 36;
		}
		else if(scale > 32)
		{
			name = "Small";
			scale = 32;
		}
		else if(scale > 29)
		{
			name = "Mini";
			scale = 29;
		}
		else if(scale > 26)
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

		// A' is generated just like A
		// rmat parameters are probabilistically symmetric
		AT.GenerateRMAT(scale, 16, 0.57, 0.19, 0.19);
				
		// relabel vertices
		SpParVec<IT,IT> p;
		RandPerm(p, AT.getlocalrows());	
		AT = AT(p,p);
		AT.PrintInfo();

		AT.RandomizeEdges();

		// TODO: this is inside the main iteration loop
		DenseParVec<int, double> y = SpMV<SR>(AT, x);
	}
	MPI::Finalize();
	return 0;
}

