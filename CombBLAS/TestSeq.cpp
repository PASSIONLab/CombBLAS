/****************************************************************/
/* Sequential and Parallel Sparse Matrix Multiplication Library */
/* version 2.2 --------------------------------------------------/
/* date: 05/11/2008 ---------------------------------------------/
/* author: Aydin Buluc (aydin@cs.ucsb.edu) ----------------------/
/* description: Sequential Test Application ---------------------/
/* tested classes: SparseDColumn --------------------------------/
/****************************************************************/

#include <iostream>
#include <fstream>
#include <string>
#include <ctime>
#include <sstream>  // Required for stringstreams
#include "SparseTriplets.h"
#include "SparseDColumn.h"
#include "randgen.h"

#ifdef MPITIMER
#include <boost/mpi.hpp>
using namespace boost::mpi;
#else
#include <boost/timer.hpp>
#endif


using namespace std;
using namespace boost;

#define NUMTYPE double
//#define CREATEFILE

#define N 10
#define NNZ 6

// creates sparse square matrices
void CreateFile(ITYPE nnz, ITYPE n, string filename)
{
	ITYPE i =1;
	ofstream out(filename.c_str());

	RandGen G;
	out<< n <<" "<< n<<" "<< nnz <<endl;
	while(i<= nnz)
	{
		out << (G.RandInt(n)+1) <<" "<<  (G.RandInt(n)+1) << " " << i << endl;
		i++;
	}
	out.close();	
}


SparseDColumn<NUMTYPE> LoadDColumn(ifstream & input)
{
	ITYPE m,n,nnz;
	input >> m >> n >> nnz;

	SparseTriplets<NUMTYPE> s(nnz,m,n);
	input >> s;

	s.SortColBased();
	
	SparseDColumn<NUMTYPE> d(s, false);

	cout<<"Converted to SparseDColumn"<<endl;
	return d;
}


int main(int argc, char* argv[])
{
	if(argc < 4)
	{
		cout << "Usage: ./testseq <BASEADDRESS> <NUMMATRICES> <NUMITER>" << endl;
		cout << "Example: ./testseq debug 4 10" << endl;
		cout << "Input files input1_p and input2_p should be under <BASEADDRESS>/proc_p/ " << endl;
		return -1;
	} 
	
	clock_t start,finish;
	double time1;
	int NUMOFMATRICES = atoi(argv[2]);
	int NUMITERATIONS = atoi(argv[3]);
	
	string rfilename = "result.txt";
	int current_itr = 0; 
	while(current_itr < NUMOFMATRICES)
	{
		ofstream outputr(rfilename.c_str()); 

		stringstream ss;
		string rank;
		ss << current_itr;
		ss >> rank;

		string directory(argv[1]);
		string subdir = "/proc";
		directory += subdir;
		directory += rank; 

		string ifilename1 = directory+"/input1_"+rank;
		string ifilename2 = directory+"/input2_"+rank;

#ifdef CREATEFILE
		CreateFile(NNZ, N, ifilename1);
		CreateFile(NNZ, N, ifilename2);
#endif

		ifstream input1(ifilename1.c_str());
		ifstream input2(ifilename2.c_str());

		
		SparseDColumn<NUMTYPE> dcol1 = LoadDColumn(input1);
		SparseDColumn<NUMTYPE> dcol2 = LoadDColumn(input2);

		// multiply them
		SparseDColumn<NUMTYPE> dcol3 = dcol1*dcol2;
		
		timer t2 = timer();
		for(int i=0;i<NUMITERATIONS; i++)
		{
			dcol3 = dcol1*dcol2;

		}
		double fin = t2.elapsed();	// wall-clock time
		
		cout<<fin<<endl;

		SparseTriplets<NUMTYPE> * tcol3 = new SparseTriplets<NUMTYPE>(dcol3);

		tcol3->SortColBased();
		cout << "Printing the result of my multiplication dcol1*dcol2 in Triplets format:" <<endl;
		outputr << *tcol3 << endl;
		delete tcol3;
		

		cout<<current_itr<<" finished!" << endl;
		current_itr++;

		input1.close();		
		input1.clear();
		
		input2.close(); 	
		input2.clear(); 

	}	
	return 0;
}

