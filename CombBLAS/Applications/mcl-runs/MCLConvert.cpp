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

void convert(string fname, vector<int64_t> mclorder, int64_t nclust = 0, string sort = "revsize")
{
    ifstream infile (fname);
    int64_t item, clustID;
    vector<vector<int64_t>> clusters;
    if(nclust > 0) clusters.resize(nclust);
    if (infile.is_open())
    {
        infile >> item >> clustID; // get rid of the header;
        while(infile >> item >> clustID)
        {
            //if(base==0)
            //    item ++; // 1-based item indexing to match MCL
            //else clustID--; // for 0-based indexing of the vector
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
    
    bool reorder = false;
    if(mclorder.size()>0) reorder = true;
    
    string outname = fname + ".mcl";
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
                    if(reorder)
                        outfile << mclorder[clusters[cl][j]] << "\t";
                    else
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
                    if(reorder)
                        outfile << mclorder[clusters[i][j]] << "\t";
                    else
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
    int nclust = 0;
    string mtxfile = "";
    
    if(argc < 2)
    {
        cout << "Usage: ./mclconvert -M <FILENAME_Output_HipMCL> (required)\n";
        cout << "-nclust <Number of clusters> (default:0)\n";
        cout << "-sort <Sort clusters by their sizes> (default:revsize)\n";
        cout << "-mtxfile <Matrix market file (used only if isolated vertices are removed by HipMCL)> \n";
        cout << "Example (0-indexed 100 clusters): ./mclconvert -M input.mtx -base 0 -nclust 100" << endl;
        return -1;
    }
    
    for (int i = 1; i < argc; i++)
    {
        if (strcmp(argv[i],"-M")==0)
        {
            ifilename = string(argv[i+1]);
            printf("filename: %s",ifilename.c_str());
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
        else if (strcmp(argv[i],"-mtxfile")==0)
        {
            mtxfile= string(argv[i + 1]);
            printf("\nMatrix market file (used to order vertices to match MCL) :%s",mtxfile.c_str());
        }
    }
    printf("\n");
    vector<int64_t> mclorder;
    if(mtxfile!="")
    {
        mclorder = MCLOrder(mtxfile);
    }
    convert(ifilename, mclorder, nclust, sort);
    return 0;
}
