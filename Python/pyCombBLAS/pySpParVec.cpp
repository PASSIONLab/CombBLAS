#include <mpi.h>

#include <iostream>
#include "pySpParVec.h"

using namespace std;

pySpParVec::pySpParVec()
{
}


//pySpParVec::pySpParVec(const pySpParMat& commSource): v(commSource.A.commGrid);
//{
//}

//pySpParVec::pySpParVec(SpParVec<int64_t, int64_t> & in_v): v(in_v)
//{
//}


int64_t pySpParVec::getnnz() const
{
	return v.getnnz();
}

const pySpParVec& pySpParVec::add(const pySpParVec& other)
{
	v.operator+=(other.v);

	return *this;
}

void pySpParVec::SetElement(int64_t index, int64_t numx)	// element-wise assignment
{
	v.SetElement(index, numx);
}


const pySpParVec& pySpParVec::subtract(const pySpParVec& other)
{
	return *this;
}

const pySpParVec& pySpParVec::invert() // "~";  almost equal to logical_not
{
	return *this;
}

const pySpParVec& pySpParVec::abs()
{
	return *this;
}

bool pySpParVec::anyNonzeros() const
{
	return false;
}

bool pySpParVec::allNonzeros() const
{
	return false;
}

int64_t pySpParVec::intersectSize(const pySpParVec& other)
{
	return 0;
}

	
void pySpParVec::load(const char* filename)
{
	ifstream input(filename);
	v.ReadDistribute(input, 0);
	input.close();
}


pySpParVec* pySpParVec::zeros(int64_t howmany)
{
	pySpParVec* ret = new pySpParVec();
	return ret;
}

pySpParVec* pySpParVec::range(int64_t howmany, int64_t start)
{
	pySpParVec* ret = new pySpParVec();
	ret->v.iota(howmany, start);
	return ret;
}

pySpParVec* EWiseMult(const pySpParVec& a, const pySpParVec& b, bool exclude)
{
	pySpParVec* ret = new pySpParVec();
	//ret->v = ::EWiseMult(a.v, b.v, exclude);
	return ret;
}

pySpParVec* EWiseMult(const pySpParVec& a, const pyDenseParVec& b, bool exclude)
{
	pySpParVec* ret = new pySpParVec();
	ret->v = ::EWiseMult(a.v, b.v, exclude, (int64_t)0);
	return ret;
}























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
			scale = 17;	// fits even to single processor
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
			// A.GenGraph500Data(initiator, scale, 16.0 * std::pow(2.0, (double)scale));	
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
		SpParVec<int64_t, int64_t> Cands = ColSums.FindInds(bind2nd(greater<int>(), 2));	
		delete AInt;	// save memory	

		SpParVec<int64_t,int64_t> RandVec, First64;
		RandPerm(RandVec,Cands.getlocnnz());	// returns 1-based permutation indices
		First64.iota(64, 1);
		Cands = Cands(RandVec(First64));
		SpParHelper::Print("Starting vertices are chosen\n");
		Cands.PrintInfo();

		for(int i=0; i<64; ++i)
		{
			// use identity (undiscovered) = 0, because 
			// (A) vertex indices that are stored in fringe are 1-based
			// (B) semantics are problematic with other values, i.e. operator+= 
			DenseParVec<int64_t, int64_t> parents ( A.getcommgrid(), (int64_t) 0);	// identity is 0 
			DenseParVec<int64_t, int> levels;
			int64_t level = 1;
			SpParVec<int64_t, int64_t> fringe;	// numerical values are stored 1-based
			fringe.SetElement(Cands[i], Cands[i]);	
			while(fringe.getnnz() > 0)
			{
				SpParVec<int64_t, int64_t> fringe = SpMV<SR>(A, fringe);	// SpMV with sparse vector
				fringe = EWiseMult(fringe, parents, true, (int64_t) 0);		// clean-up vertices that already has parents 
				parents += fringe;

				// following steps are only for validation later
				SpParVec<int64_t, int> thislevel(fringe);
				thislevel.Apply(set<int>(level+1));	
				levels += thislevel;
			}
		}
	}
	MPI::Finalize();
	return 0;
}

// comm grid
