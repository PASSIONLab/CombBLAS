#include <mpi.h>
#include <sys/time.h> 
#include <iostream>
#include <stdio.h>
#include <functional>
#include <algorithm>
#include <vector>
#include <sstream>
#ifdef NOTR1
        #include <boost/tr1/tuple.hpp>
#else
        #include <tr1/tuple>
#endif
#include "../CombBLAS/SpParVec.h"
#include "../CombBLAS/SpTuples.h"
#include "../CombBLAS/SpDCCols.h"
#include "../CombBLAS/SpParMat.h"
#include "../CombBLAS/DenseParMat.h"
#include "../CombBLAS/DenseParVec.h"


using namespace std;

// Simple helper class for declarations: Just the numerical type is templated 
// The index type and the sequential matrix type stays the same for the whole code
// In this case, they are "int" and "SpDCCols"
template <class NT>
class PSpMat 
{ 
public: 
	typedef SpDCCols < int, NT > DCCols;
	typedef SpParMat < int, NT, DCCols > MPI_DCCols;
};

/*

// Value type used by the BFS vector. If all one cares about is the depth then a simple int or double suffices.
// For graph500, though, we also need the parent pointer.
class BFSvalue {
public:
	BFSvalue(): depth(INT_MAX), parent(-1) {}
	BFSvalue(int d, int p): depth(d), parent(p) {}
	BFSvalue(const BFSvalue& other) { *this = other; }
	
public:
	int depth;  // bfs level
	int parent; // parent of this vertex in bfs tree
	
public:
	const BFSvalue& operator=(const BFSvalue& other) {
		depth = other.depth;
		parent = other.parent;
		return *this;
	}
	
	int operator<(const BFSvalue& other) {
		return depth < other.depth;
	}
};

*/

typedef struct BFSSRing_s
{
	static int add(const int arg1, const int arg2)
	{
		return std::min<int>(arg1, arg2);
	}
	
	static int multiply(const bool arg1, const int arg2)
	{
		int inf = std::numeric_limits<int>::max();
		if (arg2 == inf)
			return inf;
		
		return arg1 + arg2;
	}
} BFSSRing;

int main(int argc, char* argv[])
{
	MPI::Init(argc, argv);
	int nprocs = MPI::COMM_WORLD.Get_size();
	int myrank = MPI::COMM_WORLD.Get_rank();

	int num_starts = 3;

	if(argc < 2)
	{
		if(myrank == 0)
		{
			cout << "Usage: ./MultTest <MatrixA>" << endl;
			cout << "<MatrixA> file should be in triples format" << endl;
		}
		MPI::Finalize(); 
		return -1;
	}				
	{
		string Aname(argv[1]);		

		ifstream inputA(Aname.c_str());

		MPI::COMM_WORLD.Barrier();
	
		//typedef MinPlusSRing<bool, int> PTBOOLINT;	

		PSpMat<int>::MPI_DCCols A;
		PSpMat<int>::MPI_DCCols B, C, C2;	// construct objects
		
		SpParHelper::Print("Reading data...\n");

		A.ReadDistribute(inputA, 0);

		SpParHelper::Print("Data read\n");
		
		//C = Mult_AnXBn_Synch<PTDOUBLEDOUBLE>(A, B);
		//C = Mult_AnXBn_Synch<PTDOUBLEDOUBLE>(A, B);
		//SpParHelper::Print("Warmed up for Synch\n");
		
		MPI::COMM_WORLD.Barrier();
		double total1 = MPI::Wtime(); 	// initilize (wall-clock) timer
	
		for(int i=0; i < num_starts; i++)
		{
			//////// create starting vertices
			int startVert = -1;
			char tmpfilename[] = "g500tempstartverts.mtx";
			if (myrank == 0) {
				startVert = rand()%A.getncol();
				
				ofstream tmpout;
				tmpout.open(tmpfilename);
				
				// rows columns nnz
				tmpout << A.getncol() << " " << nprocs << " " << 1 << endl;
				tmpout << (i+1) << " " << startVert << " " << 1 << endl;;
					
				tmpout.close();
			}
	
			MPI::COMM_WORLD.Barrier();

			// read them back in
			ifstream inputB(tmpfilename);
			B.ReadDistribute(inputB, 0);
			
			if (myrank == 0)
				cout << "On starting vertex " << i << " index(" << startVert << ")" << endl;

			// start timer			
			MPI::COMM_WORLD.Barrier();
			double t1 = MPI::Wtime(); 	// initilize (wall-clock) timer

			int old_nnz = 1;
			int new_nnz = 1;
			C = Mult_AnXBn_Synch<BFSSRing>(A, B);
			new_nnz = C.getnnz();
			
			for (int j = 0; new_nnz > old_nnz && new_nnz != A.getncol(); j++) {
				old_nnz = new_nnz;
				if(myrank == 0)
					cout << "did frontier " << (j+1) << ". nnz=" << old_nnz << endl;
					
				if (j % 2 == 0) {
					C2 = Mult_AnXBn_Synch<BFSSRing>(A, C);
					new_nnz = C2.getnnz();
				} else {
					C = Mult_AnXBn_Synch<BFSSRing>(A, C2);
					new_nnz = C.getnnz();
				}
			}
			
			// final time
			MPI::COMM_WORLD.Barrier();
			double t2 = MPI::Wtime(); 	
			if(myrank == 0)
			{
				printf("%.6lf seconds elapsed \n", (t2-t1));
			}
		}		
		MPI::COMM_WORLD.Barrier();
		double total2 = MPI::Wtime(); 	
		if(myrank == 0)
		{
			printf("%.6lf seconds elapsed IN TOTAL, %.6lf seconds per BFS \n", (total2-total1), (total2-total1)/(double)num_starts);
		}

		inputA.clear();
		inputA.close();
	}
	MPI::Finalize();
	return 0;
}

