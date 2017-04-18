#include <iostream>
#include <fstream>
#include <vector>
#include <numeric>
#include <algorithm>
#include <string.h>
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

void convert(string fname, int64_t nclust = 0, int base = 1, string sort = "revsize")
{
    ifstream infile (fname);
    string line;
    int64_t item, clustID;
    vector<vector<int64_t>> clusters;
    if(nclust > 0) clusters.resize(nclust);
    if (infile.is_open())
    {
        infile >> item >> clustID; // get rid of the header;
        while(infile >> item >> clustID)
        {
            if(base==0)
                item ++; // 1-based item indexing to match MCL
            else clustID--; // for 0-based indexing of the vector
            nclust = max(nclust, clustID+1);
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
        cout << "Unable to open " << fname << endl;
        return;
    }
    
    
    string outname = fname + ".out";
    ofstream outfile (outname);
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
                for(int64_t j=0; j<clusters[cl].size() ; j++)
                {
                    outfile << clusters[cl][j] << "\t";
                }
                outfile << endl;
            }
            outfile.close();
        }
        else cout << "Unable to open " << outname << endl;
    }
    else
    {
        if (outfile.is_open())
        {
            for(int64_t i=0; i<nclust; i++)
            {
                for(int64_t j=0; j<clusters[i].size() ; j++)
                {
                    outfile << clusters[i][j] << "\t";
                }
                outfile << endl;
            }
            outfile.close();
        }
        else cout << "Unable to open " << outname << endl;
    }
    
  
    
    
}

int main(int argc, char* argv[])
{
    
    string ifilename = "";
    string sort = "revsize";
    int base = 1;
    int nclust = 0;
    
    if(argc < 2)
    {
        cout << "Usage: ./ConverCC -M <FILENAME_Output_HipMCL> (required)\n";
        cout << "-base <Starting index of clusters and items> (default:1)\n";
        cout << "-nclust <Number of clusters> (default:0)\n";
        cout << "-sort <Sort clusters by their sizes> (default:revsize)\n";
        cout << "Example (1-indexed 100 clusters): ./cc -M input.mtx -base 1 -nclust 100" << endl;
        return -1;
    }
    
    for (int i = 1; i < argc; i++)
    {
        if (strcmp(argv[i],"-M")==0)
        {
            ifilename = string(argv[i+1]);
            printf("filename: %s",ifilename.c_str());
        }
        else if (strcmp(argv[i],"-base")==0)
        {
            base = atoi(argv[i + 1]);
            printf("\nStarting index of clusters and items (1 or 0):%d",base);
        }
        else if (strcmp(argv[i],"-nclust")==0)
        {
            nclust = atoi(argv[i + 1]);
            printf("\nNumber of clusters? :%d",nclust);
        }
        else if (strcmp(argv[i],"-sort")==0)
        {
            sort = string(argv[i + 1]);
            printf("\nSorting clusters by their size (revsize or none)? :%s",sort);
        }
    }
    printf("\n");
    convert(ifilename, nclust, base, sort);
    return 0;
}
