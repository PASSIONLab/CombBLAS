#include <iostream>
#include <string>
#include <fstream>
#include <sstream>
#include <vector>
#include <algorithm>
#include <chrono>
#include <functional>
#include <cmath>
#include <map>
#include <tuple>
#include <cstdlib>
#include <cstdio>
#include <limits>
#include <omp.h>
#include "ThreadedMMConverter.h"

using namespace std;

int main(int argc, char* argv[])
{
	if(argc < 3)
    {
        cout << "Usage: ./mcl2mtx <input> <permute:0/1>" << endl;
        return 0;
    }
    
    vector<uint32_t> allrows;
    vector<uint32_t> allcols;
    vector<float> allvals;
    uint32_t nvertices;
    
    cout << "reading input matrix in MCL format... " << endl;
    
    string dictname = "Vertex_Dict_";
    dictname += string(argv[1]);
    ofstream dictout(dictname);
    
    double time_start = omp_get_wtime();
    ThreadedMMConverter(string(argv[1]), allrows, allcols, allvals, nvertices, dictout);
    cout << "ThreadedMMReader read/permuted/converted in " << omp_get_wtime() - time_start << "  seconds"<< endl;
    
    size_t nnz = allvals.size();

    string name = "Renamed_";
    name += string(argv[1]);
    ofstream out(name);
    
    time_start = omp_get_wtime();
    out << "%%MatrixMarket matrix coordinate real symmetric\n";
    out << nvertices << "\t" << nvertices << "\t" << nnz << "\n";
    for(size_t k=0; k< nnz; ++k)
    {
		out << allrows[k] << "\t" << allcols[k] << "\t" << allvals[k] << "\n";
	}
	out.close();
    cout << "Output written in " << omp_get_wtime() - time_start << "  seconds"<< endl;

	
    
}
