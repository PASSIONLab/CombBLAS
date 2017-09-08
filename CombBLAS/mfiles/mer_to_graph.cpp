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
#include <cassert>

using namespace std;

map<string, int> vertexmap; 
uint32_t numlines;


void ParseSplint(string locline, uint32_t & vertexid, ofstream & out)
{
	string name;
    	string splint;
	std::tuple<string, string, int64_t> triple;	
	
	stringstream ssline(locline);
	ssline >> name;
	assert((name == "SPLINT") || (name == "SPAN") );
	ssline >> splint;
	vector<string> contigpair(2);
	size_t pos = splint.find("<=>");
	contigpair[1] = splint.substr(pos+3);
	contigpair[0] = splint.substr(0, pos);

	pos = contigpair[0].find(".");
	contigpair[0] = contigpair[0].substr(0,pos);
	pos = contigpair[1].find(".");
	contigpair[1] = contigpair[1].substr(0,pos);

	auto ret = vertexmap.insert(make_pair(contigpair[0], vertexid));

	if (ret.second)	// successfully inserted
		++vertexid;

	get<0>(triple) = ret.first->first;
	ret = vertexmap.insert(make_pair(contigpair[1], vertexid));
	if (ret.second) ++vertexid;
	    
	get<1>(triple) = ret.first->first;
	get<2>(triple) = 1;	

	out << get<0>(triple) << "\t" << get<1>(triple) << "\t" << get<2>(triple) << "\n";
	numlines++;
}

int main(int argc, char* argv[])
{
    if(argc < 4)
    {
        cout << "Usage: ./mer2gr <splint_file> <kmer_file> <span_file" << endl;
        return 0;
    }
    
    numlines = 0;
    uint32_t vertexid = 0;
    string locline;
    ofstream out("contig_graph.txt");

    ifstream input1(argv[1]);    
    while(getline(input1,locline))
    {
	    ParseSplint(locline, vertexid, out);
    }
    input1.close();

    ifstream input2(argv[2]);    
    while(getline(input2,locline))
    {
	    ParseSplint(locline, vertexid, out);
    }
    input2.close();
    
    ifstream input3(argv[3]);  
    while(getline(input3,locline))
    {
	    ParseSplint(locline, vertexid, out);
    }
    input3.close();


    ofstream dictout("vertex_dict.txt");
    int max = 0;
    for (auto it = vertexmap.begin(); it != vertexmap.end(); ++it)
    {
	max = std::max(max, it->second);
	dictout << it ->first << endl;
    }
    cout << (max+1) << "\t" << (max+1) << "\t" << numlines << endl;
    dictout.close();

}
