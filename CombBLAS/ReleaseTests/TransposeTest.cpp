#include <mpi.h>
#include <sys/time.h> 
#include <iostream>
#include <functional>
#include <algorithm>
#include <vector>
#include <sstream>
#include <tr1/tuple>
#include "../SpParVec.h"
#include "../SpTuples.h"
#include "../SpDCCols.h"
#include "../SpParMPI2.h"
#include "../DenseParMat.h"
#include "../DenseParVec.h"


using namespace std;


int main(int argc, char* argv[])
{
	MPI::Init(argc, argv);
	int nprocs = MPI::COMM_WORLD.Get_size();
	int myrank = MPI::COMM_WORLD.Get_rank();

	if(argc < 2)
	{
		if(myrank == 0)
		{
			cout << "Usage: ./TransposeTest <BASEADDRESS>" << endl;
			cout << "Input file input.txt should be under <BASEADDRESS> in triples format" << endl;
		}
		MPI::Finalize(); 
		return -1;
	}				
	{
		string directory(argv[1]);		
		string ifilename = "input.txt";
		ifilename = directory+"/"+ifilename;

		ifstream input(ifilename.c_str());
		MPI::COMM_WORLD.Barrier();
	
		typedef SpParMPI2 <int, bool, SpDCCols<int,bool> > PARBOOLMAT;

		PARBOOLMAT A, AT;			// construct object
		A.ReadDistribute(input, 0);		// read it from file, note that we use the transpose of "input" data
		AT = A;
		AT.Transpose();
	
		A.PrintInfo();
		AT.PrintInfo();

		input.clear();
		input.close();
	}
	MPI::Finalize();
	return 0;
}
