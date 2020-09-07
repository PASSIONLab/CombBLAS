#include <iostream>
#include <fstream>
#include <vector>
#include <numeric>
#include <algorithm>
#include <string.h>
#include <assert.h>
using namespace std;

template <typename T>
vector<size_t> sortIndices(const vector<T> &v) {
    
    // initialize original index locations
    vector<size_t> idx(v.size());
    iota(idx.begin(), idx.end(), 0);
    
    // sort indexes based on comparing values in v
    sort(idx.begin(), idx.end(),
         [&v](size_t i1, size_t i2) {return v[i1] > v[i2];});
    
    return idx;
}


// simply order vertices by considering isolated vertices
vector<int64_t> MCLOrder(string fname)
{
    ifstream mtxfile (fname);
    string line;
    int64_t m, n, nnz;
    int64_t v1, v2;
    double val;
    vector<int64_t> mclorder;
    if (mtxfile.is_open())
    {
        while(mtxfile.peek()=='%') getline (mtxfile,line); // ignore comments
        mtxfile >> m >> n >> nnz;
        assert(m==n);
        vector<bool> nonisolated (m,false);
        int64_t count = 0;
        while( mtxfile >> v1 >> v2 >> val)
        {
            if(!nonisolated[v1]) {nonisolated[v1] = true; count++;}
            if(!nonisolated[v2]) {nonisolated[v2] = true; count++;}
        }
        mtxfile.close();
        
        
        cout << "vertices: " << m << " and non isolated vertices: " << count << endl;
        if(m != count)
        {
            mclorder.resize(count);
            for(int64_t i=0, j=0; i<m ; i++)
            {
                if(nonisolated[i])
                    mclorder[j++] = i;
            }
            
        }
    }
    return mclorder; // empty here
    
}

void convert(string ifname, string ofname, string sort = "revsize")
{
    ifstream infile (ifname);
    int64_t item, clustID;
    int64_t nclust = 0;
    vector<vector<int64_t>> clusters;
    if (infile.is_open())
    {
        infile >> item >> clustID; // get rid of the header;
        while(infile >> item >> clustID)
        {
            nclust = max(nclust, clustID+1); // because clustID is zero-based
            if(clustID >= clusters.size())
            {
                clusters.resize(clustID * 2 + 1);
            }
            clusters[clustID].push_back(item);
        }
        infile.close();
    }
    else
    {
        cout << "Unable to open " << ifname << endl;
        return;
    }
    
     cout << "Number of clusters from HipMCL: " << nclust << endl;
    
    ofstream outfile (ofname);
    if(sort == "revsize")
    {
        vector<int64_t> clustSize(nclust, 0);
        for(int64_t i=0; i< nclust; i++)
            clustSize[i] = clusters[i].size();
        vector<size_t> sidx = sortIndices(clustSize);
        if (outfile.is_open())
        {
            for(int64_t i=0; i<nclust; i++)
            {
                int64_t cl = sidx[i];
                for(uint64_t j=0; j<clusters[cl].size() ; j++)
                {
                    outfile << clusters[cl][j] << "\t";
                }
                outfile << endl;
            }
            outfile.close();
        }
        else cout << "Unable to open " << ofname << endl;
    }
    else
    {
        if (outfile.is_open())
        {
            for(int64_t i=0; i<nclust; i++)
            {
                for(uint64_t j=0; j<clusters[i].size() ; j++)
                {
                    outfile << clusters[i][j] << "\t";
                }
                outfile << endl;
            }
            outfile.close();
        }
        else cout << "Unable to open " << ofname << endl;
    }
    
    
    
}

int main(int argc, char* argv[])
{
    
    string ifilename = "";
    string ofilename = "";
    string sort = "revsize";
    
    cout << "Reformatting HipMCL output to MCL format.....\n";
    if(argc < 4)
    {
        cout << "Usage: ./mclconvert -M <IN_FILENAME> -o <OUT_FILENAME>(required)\n";
        cout << "-sort <Sort clusters by their sizes> (default:revsize)\n";
        cout << "Example: ./mclconvert -M input.mtx" << endl;
        return -1;
    }
    
    for (int i = 1; i < argc; i++)
    {
        if (strcmp(argv[i],"-M")==0)
        {
            ifilename = string(argv[i+1]);
            printf("Input filename: %s",ifilename.c_str());
        }
        else if (strcmp(argv[i],"-o")==0)
        {
            ofilename = string(argv[i+1]);
            printf("Output filename: %s",ofilename.c_str());
        }
        else if (strcmp(argv[i],"-sort")==0)
        {
            sort = string(argv[i + 1]);
            printf("\nSorting clusters by their size (revsize or none)? :%s",sort.c_str());
        }
    }
    printf("\n");
    convert(ifilename, ofilename, sort);
    return 0;
}
