#include <iostream>
#include <string>
#include <fstream>
#include <sstream>
#include <vector>
#include <algorithm>
#include <chrono>
#include <functional>
#include <cmath>
#include <boost/algorithm/string.hpp>
#include <map>
#include <tuple>
#include <cstdlib>
#include <cstdio>
#include <limits>
#include <omp.h>

using namespace std;

int main(int argc, char* argv[])
{
	if(argc < 2)
        {
                cout << "Usage: ./renamer <input>" << endl;
                return 0;
        }
        ifstream input1(argv[1], ifstream::in);
	map<string, int> vertexmap;
	int vertexid = 0;

	string name = "Renamed_";
	name += string(argv[1]);
	ofstream out(name);

	string locline;
	int numlines = 0;
        while(getline(input1,locline))
        {
		std::tuple<int64_t, int64_t, string> triple;	// the third entry is a float, but that doesn't matter here
                vector<string> strs;
                boost::split(strs, locline, boost::is_any_of("\t "));

		auto ret = vertexmap.insert(make_pair(strs[0], vertexid));
		if (ret.second)	// successfully inserted
		++vertexid;

		// map::insert returns a pair, with its member pair::first set to an 
		// iterator pointing to either the newly inserted element or to the element with an equivalent key in the map
			
		get<0>(triple) = ret.first->second;
			
		ret = vertexmap.insert(make_pair(strs[1], vertexid));
                if (ret.second) ++vertexid;

		get<1>(triple) = ret.first->second;
		get<2>(triple) = strs[2];	
		numlines++;

		out << get<0>(triple) << "\t" << get<1>(triple) << "\t" << get<2>(triple) << "\n";
	}
	out.close();
	string dictname = "Vertex_Dict_";
	dictname += string(argv[1]);
	ofstream dictout(dictname);
	int max = 0;
	for (auto it = vertexmap.begin(); it != vertexmap.end(); ++it)
	{
		max = std::max(max, it->second);
		dictout << it->second << "\t" << it ->first << endl;
	}
	cout << (max+1) << "\t" << (max+1) << "\t" << numlines << endl;
	dictout.close();
}
