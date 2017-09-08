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
#include <boost/algorithm/string.hpp>

using namespace std;

map<string, int> vertexmap; 
uint32_t numlines;


void ParseSplint(string locline, uint32_t & vertexid, ofstream & out)
{
	string name;
    	string splint;
	std::tuple<int64_t, int64_t, int64_t> triple;	
	
	stringstream ssline(locline);
	ssline >> name;
	assert((name == "SPLINT") || (name == "SPAN") );
	ssline >> splint;
	vector<string> contigpair;
	boost::split(contigpair, splint, boost::is_any_of("<=>"));
	if(contigpair[0].length() < 1 && contigpair[1].length() < 1)
	{
		cout << "Issue in line " << locline << endl;
	}
	auto ret = vertexmap.insert(make_pair(contigpair[0], vertexid));

	if (ret.second)	// successfully inserted
		++vertexid;

	get<0>(triple) = ret.first->second;
	ret = vertexmap.insert(make_pair(contigpair[1], vertexid));
	if (ret.second) ++vertexid;
	    
	get<1>(triple) = ret.first->second;
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
	dictout << it->second << "\t" << it ->first << endl;
    }
    cout << (max+1) << "\t" << (max+1) << "\t" << numlines << endl;
    dictout.close();

}
