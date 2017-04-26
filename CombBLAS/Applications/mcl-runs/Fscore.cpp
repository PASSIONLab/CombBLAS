#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <numeric>
#include <algorithm>
#include <string.h>
#include <assert.h>
#include <omp.h>
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

int64_t intersect(vector<int64_t> vec1, vector<int64_t> vec2)
{
    int64_t isect = 0;
    for(int i=0, j=0; i<vec1.size() && j<vec2.size();)
    {
        if(vec1[i] < vec2[j]) i++;
        else if(vec2[j] < vec1[i]) j++;
        else
        {
            isect ++;
            i++; j++;
        }
    }
    return isect;
}

// fname1 is the reference (form MCL)
double Fscore(string fname1, string fname2, int base)
{
    ifstream infile1 (fname1);
    ifstream infile2 (fname2);
    string line;
    int64_t item, clustID = 0;
    int64_t nclust1, nclust2;
    int64_t N1 = 0, N2 = 0; // total items
    // item 0-based
    
    vector<vector<int64_t>> clusters1;
    
    if (infile1.is_open())
    {
        while ( getline (infile1,line) )
        {
            istringstream iss(line);
            if(clustID >= clusters1.size())
            {
                clusters1.resize(clustID * 2 + 1);
            }
            while ( iss >> item)
            {
                if(base==1) item --;
                clusters1[clustID].push_back(item);
            }
            
            N1 += clusters1[clustID].size();
            clustID++;
        }
        infile1.close();
        nclust1 = clustID;
    }
    else
    {
        cout << "Unable to open " << fname1 << endl;
        return -1;
    }
    
    vector<int64_t> clust2(N1, -1);
    clustID = 0;
    if (infile2.is_open())
    {
        while ( getline (infile2,line) )
        {
            istringstream iss(line);
            while ( iss >> item)
            {
                if(base==1) item --;
                clust2[item] = clustID;
            }
            clustID++;
        }
        infile2.close();
        nclust2 = clustID;
    }
    else
    {
        cout << "Unable to open " << fname2 << endl;
        return -1;
    }
    
    vector<int64_t> clusterSizes2(nclust2,0);
    for(int i=0; i<N1; i++)
    {
        clusterSizes2[clust2[i]]++;
    }
    
    
    vector<double> F(nclust1);
    vector<int64_t> nisect(nclust2); // number of items in the intersection
    vector<int64_t> isect(nclust2); // clusters with nonzero intersection with the current cluster
    

    for(int i=0; i<nclust1; i++)
    {
        int64_t isectCount = 0;
        fill(nisect.begin(), nisect.end(), 0);
        for(int j=0; j<clusters1[i].size(); j++)
        {
            int64_t item1 = clusters1[i][j];
            int64_t c2 = clust2[item1];
            if(nisect[c2]==0) isect[isectCount++] = c2;
            nisect[c2] ++;
        }
        double Fi = 0;
        for(int j=0; j<isectCount; j++)
        {
            int64_t c2 = isect[j];
            double precision = (double) nisect[c2] / clusterSizes2[j]; //
            double recall = (double) nisect[c2] / clusters1[i].size();
            double Fij = 2 * precision * recall / (precision + recall);
            Fi = max(Fi, Fij);
        }
        
        Fi = Fi * clusters1[i].size()/ N1;
        F[i] = Fi;
    }
    return accumulate(F.begin(), F.end(), 0.0);
}

int main(int argc, char* argv[])
{
    
    string ifilename1 = "";
    string ifilename2 = "";
    int base = 0;
  
    
    if(argc < 2)
    {
        cout << "Usage: ./fscore -M1 <FILENAME_Output_ MCL> -M2 <FILENAME_Output_HipMCL> -base <Base of indices 1 or 0>\n";
        cout << "Example: ./fscore -M1 input1.txt -M2 input2.txt -base 0 " << endl;
        return -1;
    }
    
    for (int i = 1; i < argc; i++)
    {
        if (strcmp(argv[i],"-M1")==0)
        {
            ifilename1 = string(argv[i+1]);
            printf("\nfilename1: %s",ifilename1.c_str());
        }
        else if (strcmp(argv[i],"-M2")==0)
        {
            ifilename2 = string(argv[i+1]);
            printf("\nfilename2: %s",ifilename2.c_str());
        }
        else if (strcmp(argv[i],"-base")==0)
        {
            base = atoi(argv[i+1]);
            printf("\nbase: %d",base);
        }
    }
    printf("\n");
    double F = Fscore(ifilename1, ifilename2, base);
    cout << "F score: " << F << endl;
    return 0;
}
