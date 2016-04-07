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
#include "MMConverter.h"

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

    string name = "Renamed_";
    name += string(argv[1]);
    ofstream out(name);
    
    MMConverter(string(argv[1]), dictout, out);
    dictout.close();
    out.close();
}
