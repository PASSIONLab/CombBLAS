#include <iostream>
#include <vector>

#include "CombBLAS/CombBLAS.h"

using namespace std;
using namespace combblas;

typedef int64_t IT;
typedef double NT;



int main(int argc, char* argv[])
{
	int nprocs, myrank;
	MPI_Init(&argc, &argv);
	MPI_Comm_size(MPI_COMM_WORLD,&nprocs);
	MPI_Comm_rank(MPI_COMM_WORLD,&myrank);

	if(argc < 4)
	{
		if(myrank == 0)
			cout << "Usage: ./BlockedSpGEMM <MatrixA> <MatrixB> <br> <bc>" << endl;
		MPI_Finalize(); 
		return -1;
	}

	
	{
		string Aname(argv[1]);
		string Bname(argv[2]);
		int br = atoi(argv[3]);
		int bc = atoi(argv[4]);
	
		MPI_Barrier(MPI_COMM_WORLD);
		typedef PlusTimesSRing<NT, NT> SR_PT;
		typedef SpDCCols<IT, NT> DER;
		
		shared_ptr<CommGrid> fullWorld;
		fullWorld.reset(new CommGrid(MPI_COMM_WORLD, 0, 0));
        	
		SpParMat<IT, NT, DER> A(fullWorld);            	
		A.ParallelReadMM(Aname, true, maximum<NT>());
		IT nr = A.getnrow(), nc = A.getncol(), nnz = A.getnnz();
		if (myrank == 0)
			cout << "A " << nr << " " << nc << " " << nnz << std::endl;

		SpParMat<IT, NT, DER> B(fullWorld);	
		B.ParallelReadMM(Bname, true, maximum<NT>());
		nr = B.getnrow(), nc = B.getncol(), nnz = B.getnnz();
		if (myrank == 0)
			cout << "B " << nr << " " << nc << " " << nnz << std::endl;

		// auto blocks = A.BlockSplit(br, bc);
		BlockSpGEMM<IT, NT, DER, NT, DER> bspgemm(A, B, br, bc);
		IT roffset, coffset;
		while (bspgemm.hasNext())
		{
			auto C = bspgemm.getNextBlock<SR_PT, NT, DER>(roffset, coffset);
			nr = C.getnrow(), nc = C.getncol(), nnz = C.getnnz();
			if (myrank == 0)
				cout << "block size " << nr << " " << nc << " " << nnz
					 << " offsets " << roffset << " " << coffset
					 << std::endl;
		}
		
		// auto C = bspgemm.getBlockId<SR_PT, NT, DER>(0, 1, roffset, coffset);
		// nr = C.getnrow(), nc = C.getncol(), nnz = C.getnnz();
		// if (myrank == 0)
		// 	cout << "block size " << nr << " " << nc << " " << nnz
		// 		 << " offsets " << roffset << " " << coffset
		// 		 << std::endl;
		
	}

	
	MPI_Finalize();
	return 0;
}
