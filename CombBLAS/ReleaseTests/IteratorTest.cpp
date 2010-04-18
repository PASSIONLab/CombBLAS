#include <sys/time.h> 
#include <iostream>
#include <functional>
#include <algorithm>
#include <vector>
#include <sstream>
#include "../SpDCCols.h"
#include "../SpParMat.h"

using namespace std;


int main(int argc, char* argv[])
{

	MPI::Init(argc, argv);
	int nprocs = MPI::COMM_WORLD.Get_size();
	int myrank = MPI::COMM_WORLD.Get_rank();

	if(argc < 3)
	{
		if(myrank == 0)
		{
			cout << "Usage: ./IteratorTest <BASEADDRESS> <Matrix>" << endl;
			cout << "Input file <Matrix> should be under <BASEADDRESS> in triples format" << endl;
		}
		MPI::Finalize(); 
		return -1;
	}				
	{
		string directory(argv[1]);		
		string name(argv[2]);
		name = directory+"/"+name;

		ifstream input(name.c_str());
		MPI::COMM_WORLD.Barrier();
	
		typedef SpParMat <int, double, SpDCCols<int,double> > PARMAT;

		PARMAT A;
		A.ReadDistribute(input, 0);	// read it from file

		int count = 0;	
		int total = 0;
	
		for(SpDCCols<int,double>::SpColIter colit = A.seq().begcol(); colit != A.seq().endcol(); ++colit)	// iterate over columns
		{
			for(SpDCCols<int,double>::SpColIter::NzIter nzit = A.seq().begnz(colit); nzit != A.seq().endnz(colit); ++nzit)
			{	
				// cout << nzit.rowid() << '\t' << colit.colid() << '\t' << nzit.value() << '\n';	
				count++;
			}
		}	
		MPI::COMM_WORLD.Allreduce( &count, &total, 1, MPIType<int>(), MPI::SUM);
		
		if(total == A.getnnz())
			SpParHelper::Print( "Iteration passed soft test\n");
		else
			SpParHelper::Print( "Iteration failed !!!\n") ;
		
			
		input.clear();
		input.close();
	}
	MPI::Finalize();
	return 0;
}

