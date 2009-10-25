#include <mpi.h>
#include <sys/time.h> 
#include <iostream>
#include <functional>
#include <algorithm>
#include <vector>
#include <sstream>
#include "../SpParVec.h"
#include "../SpTuples.h"
#include "../SpDCCols.h"
#include "../SpParMat.h"
#include "../DenseParMat.h"
#include "../DenseParVec.h"


using namespace std;


int main(int argc, char* argv[])
{
	MPI::Init(argc, argv);
	int nprocs = MPI::COMM_WORLD.Get_size();
	int myrank = MPI::COMM_WORLD.Get_rank();

	if(argc < 4)
	{
		if(myrank == 0)
		{
			cout << "Usage: ./TransposeTest <BASEADDRESS> <Matrix> <MatrixTranspose>" << endl;
			cout << "Input file <Matrix> and <MatrixTranspose> should be under <BASEADDRESS> in triples format" << endl;
		}
		MPI::Finalize(); 
		return -1;
	}				
	{
		string directory(argv[1]);		
		string normalname(argv[2]);
		string transname(argv[3]);
		normalname = directory+"/"+normalname;
		transname = directory+"/"+transname;

		ifstream inputnormal(normalname.c_str());
		ifstream inputtrans(transname.c_str());
		MPI::COMM_WORLD.Barrier();
	
		typedef SpParMat <int, bool, SpDCCols<int,bool> > PARBOOLMAT;

		PARBOOLMAT A, AT, ATControl;		// construct object
		A.ReadDistribute(inputnormal, 0);	// read it from file, note that we use the transpose of "input" data
		AT = A;
		AT.Transpose();

		ATControl.ReadDistribute(inputtrans, 0);
		if (ATControl == AT)
		{
			SpParHelper::Print("Transpose working correctly\n");	
		}
		else
		{
			SpParHelper::Print("ERROR in transpose, go fix it!\n");	
		}

		inputnormal.clear();
		inputnormal.close();
		inputtrans.clear();
		inputtrans.close();
	}
	MPI::Finalize();
	return 0;
}
