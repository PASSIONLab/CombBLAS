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

using namespace std;



int main(int argc, char* argv[])
{
    if(argc < 2)
    {
       cout << "Usage: ./components <components_file>" << endl;
        return 0;
    }
    
    string vname, compname;
    ifstream inputvert(argv[1]);
    map<string, int> compcounts;
    vector< pair<int, string > > countbyname;
    
    while(inputvert >> vname)
    {
	inputvert >> compname;
	auto it = compcounts.insert(make_pair(compname,1));
	if (!it.second)	// already there
	{
		(it.first->second)++;
	}
    }
    cout << "distinct components " << compcounts.size() << endl;

   	for (auto it = compcounts.begin(); it != compcounts.end(); ++it)
	{
		countbyname.push_back(make_pair(it->second, it->first));
	} 
sort(countbyname.begin(), countbyname.end());
for (auto it = countbyname.begin(); it != countbyname.end(); ++it)
{
	cout << it ->second << " " << it->first << endl; 
}
    
}
