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
#include "../SpParVec.h"
#include "../SpTuples.h"
#include "../SpDCCols.h"
#include "../SpParMat.h"
#include "../DenseParMat.h"
#include "../DenseParVec.h"
#include "../Operations.h"

using namespace std;


// Simple helper class for declarations: Just the numerical type is templated 
// The index type and the sequential matrix type stays the same for the whole code
// In this case, they are "int" and "SpDCCols"
template <class NT>
class Dist
{ 
public: 
	typedef SpDCCols < int, NT > DCCols;
	typedef SpParMat < int, NT, DCCols > MPI_DCCols;
	typedef DenseParVec < int, NT> MPI_DenseVec;
};



int main(int argc, char* argv[])
{
	MPI::Init(argc, argv);
	int nprocs = MPI::COMM_WORLD.Get_size();
	int myrank = MPI::COMM_WORLD.Get_rank();
	
	typedef PlusTimesSRing<bool, bool> PTBOOLBOOL;

	if(argc < 3)
        {
		if(myrank == 0)
		{	
                	cout << "Usage: ./apowers <BASEADDRESS> <KLIMIT>" << endl;
                	cout << "Example: ./apowers Data/ 5" << endl;
                	cout << "Input file input.txt should be under <BASEADDRESS> in triples format" << endl;
                }
		MPI::Finalize(); 
		return -1;
        }

	{
		double klimit = atof(argv[2]);

		string directory(argv[1]);		
		string ifilename = "input.txt";
		ifilename = directory+"/"+ifilename;

		ifstream input(ifilename.c_str());
		if( !input ) 
		{
		    	SpParHelper::Print( "Error opening input stream\n");
    			return -1;
  		}
		MPI::COMM_WORLD.Barrier();
	
		Dist<double>::MPI_DCCols A;	// construct object
		A.ReadDistribute(input, 0);	// read it from file
	
		input.clear();
		input.close();

		int k = 1;
		std::string s;
		std::stringstream out;
		out << k;
		s = out.str();
		A.PrintForPatoh("A"+s+".patoh");

		SpParHelper::Print("Original matrix printed...\n");
		A.PrintInfo();
		while (k < klimit)
		{
			A.Square<PTBOOLBOOL>();
			k *= 2;
			
			std::string s;
			std::stringstream out;
			out << k;
			s = out.str();
			A.PrintForPatoh("A"+s+".patoh");

			string message = "Power " + s + " printed\n"; 
			SpParHelper::Print(message);
			A.PrintInfo();
		}
	}	

	// make sure the destructors for all objects are called before MPI::Finalize()
	MPI::Finalize();	
	return 0;
}

