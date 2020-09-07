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
#define COMPRESS_STRING

int main(int argc, char* argv[])
{
    if(argc < 3)
    {
        cout << "Usage: ./mcl2mtx <full_path_to_input> <outputprefix>" << endl;
        return 0;
    }
    
    cout << "reading input matrix in MCL format... " << endl;
    
    string dictname = "Vertex_Dict_";
    dictname += string(argv[2]);
    ofstream dictout(dictname);
    
    MMConverter(string(argv[1]), dictout, string(argv[2]));
    dictout.close();
}
