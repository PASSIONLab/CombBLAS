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



template <class IT, class NT>
class PairReadSaveHandler
{
	public:
		PairReadSaveHandler() {};
		pair<NT,NT> getNoNum(IT index) { return make_pair((NT) 0, (NT) 0); }

		
		template <typename c, typename t>
		pair<NT,NT> read(std::basic_istream<c,t>& is, IT index)
		{
			pair<NT,NT> pp;
			is >> pp.first >> pp.second;
			return pp;
		}
		
	
		template <typename c, typename t>
		void save(std::basic_ostream<c,t>& os, const pair<NT,NT> & pp, IT index)	
		{
			os << pp.first << "\t" << pp.second;
		}
};


int main(int argc, char* argv[])
{
	int nprocs, myrank;
	MPI_Init(&argc, &argv);
	MPI_Comm_size(MPI_COMM_WORLD,&nprocs);
	MPI_Comm_rank(MPI_COMM_WORLD,&myrank);

	if(argc < 5)
	{
		if(myrank == 0)
		{
			cout << "Usage: ./VectorIO <DictionaryVector> <UnPermutedVector> <OutputName> base(0 or 1)" << endl;
			cout << "Example: ./VectorIO vertex_dict.mtx clusters.mtx original_clusters.mtx 0" << endl;
		}
		MPI_Finalize(); 
		return -1;
	}				
	{
		string vec1name(argv[1]);
		string vec2name(argv[2]);
		string voutname(argv[3]);
		int base = atoi(argv[4]);


		FullyDistVec<int64_t,string> vecdict, clusters;	// keep cluster ids as "string" for generality
	
		// the "index sets" of vecdict and clusters are the same
		// goal here is to just convert the indices in "clusters" to values in vecdict 
			
		vecdict.ParallelRead(vec1name, base, // keeps lexicographically larger one in case of duplicates (after warning)
				[](string s1, string s2) { cout << "Unexpected duplicate in dictionary" << endl; return std::max<string>(s1, s2); });

		clusters.ParallelRead(vec2name, base, // keeps lexicographically larger one in case of duplicates (after warning)
				[](string s1, string s2) { cout << "Unexpected duplicate in unpermuted vector" << endl; return std::max<string>(s1, s2); });	
        
		vecdict.PrintInfo("dictionary");
        	clusters.PrintInfo("unpermuted cluster vector");
              
		// FullyDistVec<IT,NT>::EWiseOut(const FullyDistVec<IT,NT> & rhs, _BinaryOperation __binary_op, FullyDistVec<IT,OUT> & result)
		// Perform __binary_op(*this[i], rhs[i]) for every element in rhs and *this, write the result output vector
		FullyDistVec<int64_t, pair<string, string> > newclusters;
		
		clusters.EWiseOut(vecdict, [] (string s1, string s2) { return make_pair(s1,s2); },  newclusters);
       
       		newclusters.ParallelWrite(voutname, base, PairReadSaveHandler<int64_t, string>(), false);	// don't print the index		
       	}
	MPI_Finalize();
	return 0;
}
