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

	if(argc < 6)
	{
		if(myrank == 0)
		{
			cout << "Usage: ./IndexingTest <BASEADDRESS> <Matrix> <IndexedMatrix> <VectorOne> <VectorTwo>" << endl;
			cout << "Example: ./IndexingTest ../mfiles B_100x100.txt B_10x30_Indexed.txt rand10outta100.txt rand30outta100.txt" << endl;
			cout << "Input files should be under <BASEADDRESS> in tuples format" << endl;
		}
		MPI::Finalize(); 
		return -1;
	}				
	{
		string directory(argv[1]);		
		string normalname(argv[2]);
		string indexdname(argv[3]);
		string vec1name(argv[4]);
		string vec2name(argv[5]);
		normalname = directory+"/"+normalname;
		indexdname = directory+"/"+indexdname;
		vec1name = directory+"/"+vec1name;
		vec2name = directory+"/"+vec2name;

		ifstream inputnormal(normalname.c_str());
		ifstream inputindexd(indexdname.c_str());
		ifstream inputvec1(vec1name.c_str());
		ifstream inputvec2(vec2name.c_str());

		if(myrank == 0)
		{	
			if(inputnormal.fail() || inputindexd.fail() || inputvec1.fail() || inputvec2.fail())
			{
				cout << "One of the input files do not exist, aborting" << endl;
				MPI::COMM_WORLD.Abort(NOFILE);
				return -1;
			}
		}
		MPI::COMM_WORLD.Barrier();
		typedef SpParMat <int, double, SpDCCols<int,double> > PARDBMAT;
		PARDBMAT A, AID, ACID;		// declare objects
		SpParVec<int,int> vec1, vec2;

		A.ReadDistribute(inputnormal, 0);	
		AID.ReadDistribute(inputindexd, 0);	
		vec1.ReadDistribute(inputvec1, 0);
		vec2.ReadDistribute(inputvec2, 0);

		vec1.Apply(bind2nd(minus<int>(), 1));	// For 0-based indexing
		vec2.Apply(bind2nd(minus<int>(), 1));	
		ACID = A(vec1, vec2);

		ACID.PrintInfo();
		if (ACID == AID)
		{
			SpParHelper::Print("Indexing working correctly\n");	
		}
		else
		{
			SpParHelper::Print("ERROR in indexing, go fix it!\n");	
		}

		// generate random permutations
		SpParVec<int,int> p, q;
		p.iota(A.getnrow(), 0);
		q.iota(A.getncol(), 0);
		RandPerm(p);	
		RandPerm(q);

		PARDBMAT B = A(p,q);
		A.PrintInfo();
		B.PrintInfo();

		float oldbalance = A.LoadImbalance();
		float newbalance = B.LoadImbalance();
		ostringstream outs;
		outs << "Old balance: " << oldbalance << endl;
		outs << "New balance: " << newbalance << endl;
		SpParHelper::Print(outs.str());
		SpParHelper::Print(outs.str());

		// B = P A Q
		// get the inverse permutations
		SpParVec<int, int> pinv = p.sort();
		SpParVec<int, int> qinv = q.sort();
		PARDBMAT C = B(pinv,qinv);
		if(C == A)
		{
			SpParHelper::Print("Double permutation successfully restored the original\n");	
		}
		else
		{
			A.PrintForPatoh("Original.patoh");
			B.PrintForPatoh("Permuted.patoh");	
			C.PrintForPatoh("Restored.patoh");
		}

		inputnormal.clear();
		inputnormal.close();
		inputindexd.clear();
		inputindexd.close();
		inputvec1.clear();
		inputvec1.close();
		inputvec2.clear();
		inputvec2.close();
	}
	MPI::Finalize();
	return 0;
}
