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
#include "DisjSets.h"

using namespace std;


// typedef void tommy_foreach_arg_func(void* arg, void* obj);

void* printfunc(void* arg, void* obj)
{
    pair<DisjSets*, ofstream*> * mypair = (pair<DisjSets*, ofstream*> *) arg;    // cast argument
    DisjSets * ds = mypair->first;
    ofstream * out = mypair->second;
    (*out) << ((tommy_object *) obj)->vname << "\t" << ds->find( (int) ((tommy_object *) obj)->vid) << "\n";
}

int main(int argc, char* argv[])
{
	if(argc < 3)
    {
        cout << "Usage: ./cc <vertices_file> <edges_file>" << endl;
        return 0;
    }
    
    ifstream inputvert(argv[1]);
    char vname[150];
    uint32_t vertexid = 0;
    
    tommy_hashdyn hashdyn;
    tommy_hashdyn_init(&hashdyn);
    while(inputvert >> vname)
    {
        string s_vname(vname);  // string version
        tommy_object* obj = new tommy_object(vertexid, s_vname);   // (vertexid,s_vname) pair is the payload (data)
        tommy_hashdyn_insert(&hashdyn, &(obj->node), obj, tommy_hash_u32(0, vname, strlen(vname))); // hashed string is key
        vertexid++;
    }
    cout << "vertex list read, there are " << vertexid << endl;
    DisjSets ds(vertexid);
    
    FILE *f;
    if ((f = fopen(argv[2], "r")) == NULL)
    {
        printf("file %s can not be found\n", argv[2]);
        exit(1);
    }
    
    // Use fseek again to go backwards two bytes and check that byte with fgetc
    struct stat st;     // get file size
    if (stat(argv[2], &st) == -1)
    {
        exit(1);
    }
    int64_t file_size = st.st_size;
    cout << "Edge file is " << file_size << " bytes" << endl;
    long int ffirst = ftell(f); // doesn't change
    long int fpos = ffirst;
    long int end_fpos = file_size;
    
    vector<string> lines;
    bool finished = FetchBatch(f, fpos, end_fpos, true, lines); // fpos will move
    int64_t entriesread = lines.size();
    
    
    char from[128];
    char to[128];
    double vv;
    for (vector<string>::iterator itr=lines.begin(); itr != lines.end(); ++itr)
    {
        // string::c_str() -> Returns a pointer to an array that contains a null-terminated sequence of characters (i.e., a C-string)
        sscanf(itr->c_str(), "%s %s %lg", from, to, &vv);
        string s_from = string(from);
        string s_to = string(to);
        
        tommy_object* obj1 = (tommy_object*) tommy_hashdyn_search(&hashdyn, compare, &s_from, tommy_hash_u32(0, from, strlen(from)));
        if(!obj1)
        {
            cout << "This doesn't make sense! " << s_from <<  " should exist" << endl;
        }
        
        tommy_object* obj2 = (tommy_object*) tommy_hashdyn_search(&hashdyn, compare, &s_to, tommy_hash_u32(0, to, strlen(to)));
        if(!obj2)
        {
            cout << "This doesn't make sense! " << s_to <<  " should exist" << endl;
        }
        int set1 = ds.find((int) obj1->vid);
        int set2 = ds.find((int) obj2->vid);
        if(set1 != set2)
        {
            ds.unionSets(set1, set2);
        }
    }
    vector<string>().swap(lines);
    
    while(!finished)
    {
        finished = FetchBatch(f, fpos, end_fpos, false, lines);
        entriesread += lines.size();
        cout << "entriesread: " << entriesread << ", current vertex id: " << vertexid << endl;
        
        // Process files
        char from[128];
        char to[128];
        double vv;
        for (vector<string>::iterator itr=lines.begin(); itr != lines.end(); ++itr)
        {
            // string::c_str() -> Returns a pointer to an array that contains a null-terminated sequence of characters (i.e., a C-string)
            sscanf(itr->c_str(), "%s %s %lg", from, to, &vv);
            
            string s_from = string(from);
            string s_to = string(to);
            
            tommy_object* obj1 = (tommy_object*) tommy_hashdyn_search(&hashdyn, compare, &s_from, tommy_hash_u32(0, from, strlen(from)));
            if(!obj1)
            {
                cout << "This doesn't make sense! " << s_from <<  " should exist" << endl;
            }
            
            tommy_object* obj2 = (tommy_object*) tommy_hashdyn_search(&hashdyn, compare, &s_to, tommy_hash_u32(0, to, strlen(to)));
            if(!obj2)
            {
                cout << "This doesn't make sense! " << s_to <<  " should exist" << endl;
            }
            int set1 = ds.find((int) obj1->vid);
            int set2 = ds.find((int) obj2->vid);
            if(set1 != set2)
            {
                ds.unionSets(set1, set2);
            }
        }
        vector<string>().swap(lines);
    }
    cout << "There are " << vertexid << " vertices and " << entriesread << " edges" << endl;
    
    string s_out(argv[1]);
    s_out += ".components";
    ofstream output(s_out);
    
    pair<DisjSets*, ofstream*> mypair(&ds, &output);
    tommy_hashdyn_foreach_arg(&hashdyn, (tommy_foreach_arg_func *) printfunc, &mypair);
    
}
