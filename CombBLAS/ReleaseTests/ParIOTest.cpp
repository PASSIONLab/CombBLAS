#include <mpi.h>
#include <sys/time.h> 
#include <iostream>
#include <functional>
#include <algorithm>
#include <vector>
#include <sstream>
#include "CombBLAS/CombBLAS.h"

using namespace std;
using namespace combblas;



// Simple helper class for declarations: Just the numerical type is templated 
// The index type and the sequential matrix type stays the same for the whole code
// In this case, they are "int" and "SpDCCols"
template <class NT>
class PSpMat 
{ 
public: 
	typedef SpDCCols < int64_t, NT > DCCols;
	typedef SpParMat < int64_t, NT, DCCols > MPI_DCCols;
};

class StdArrayReadSaveHandler
{
public:
	array<char,MAXVERTNAME> getNoNum(int64_t index) { return array<char,MAXVERTNAME>(); }
		
	template <typename c, typename t>
	array<char,MAXVERTNAME> read(std::basic_istream<c,t>& is, int64_t index)
	{
		array<char,MAXVERTNAME> strarray;
		string str;
		is >> str;	// read into str
		std::copy( str.begin(), str.end(), strarray.begin() ); 
	       	if(str.length() < MAXVERTNAME)  strarray[str.length()] = '\0'; // null terminating char	
	
		return strarray;
	}
	
	template <typename c, typename t>
	void save(std::basic_ostream<c,t>& os, const array<char,MAXVERTNAME>& strarray, int64_t index)
	{
		auto locnull = find(strarray.begin(), strarray.end(), '\0');
		string str(strarray.begin(), locnull); 
		os << str;
	}
};


int main(int argc, char* argv[])
{
	int nprocs, myrank;
	MPI_Init(&argc, &argv);
	MPI_Comm_size(MPI_COMM_WORLD,&nprocs);
	MPI_Comm_rank(MPI_COMM_WORLD,&myrank);

	if(argc < 3)
	{
		if(myrank == 0)
		{
			cout << "Usage: ./ParIOTest <MatrixA> <MatrixB_general>" << endl;
			cout << "<MatrixA> is an absolute address, and file should be in Matrix Market format" << endl;
			cout << "<MatrixB_general> is an absolute address, file is in general triples format (MCL calls this label input)" << endl;			
			
		}
		MPI_Finalize(); 
		return -1;
	}				
	{
		string Aname(argv[1]);	
		string Bname(argv[2]);
	
        	PSpMat<double>::MPI_DCCols A, B;
		
        	A.ParallelReadMM(Aname, true, maximum<double>());
		FullyDistVec<int64_t, array<char, MAXVERTNAME> > perm = B.ReadGeneralizedTuples(Bname, maximum<double>());


		if (A == B)
		{
			SpParHelper::Print("Parallel Matrix Market I/O working correctly\n");
		}
		else
		{
			SpParHelper::Print("ERROR in Parallel Matrix Market I/O");
			A.ParallelWriteMM("A_Error.mtx", true);
			B.ParallelWriteMM("B_Error.mtx", true);
		}

		perm.ParallelWrite("PermutationVec.mtx", 1, StdArrayReadSaveHandler(), true);
	}
	MPI_Finalize();
	return 0;
}

